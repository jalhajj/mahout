package org.apache.mahout.cf.taste.impl.recommender;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.eval.Fold;
import org.apache.mahout.cf.taste.impl.eval.KFoldDataSplitter;
import org.apache.mahout.cf.taste.impl.recommender.MetaRecommender.RecWrapper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MixedHybridRecommender extends AbstractRecommender {
	
	private static final Logger log = LoggerFactory.getLogger(MixedHybridRecommender.class);

	static class UserBlender {
		
		private final ArrayList<Integer> hits;
		private int cnt;
		
		UserBlender(int nbAlgos) {
			this.hits = new ArrayList<Integer>(Collections.nCopies(nbAlgos, 0));
			this.cnt = 0;
		}
		
		int getCount() {
			return this.cnt;
		}
		
		int getNbHits(int idx) {
			return this.hits.get(idx);
		}
		
		void add(int idx) {
			this.hits.set(idx, this.hits.get(idx) + 1);
			this.cnt++;
		}
		
	}
	
	private final ArrayList<RecommenderBuilder> builders;
	private final ArrayList<Recommender> recs;
	private final int nrecs;
	private final FastByIDMap<UserBlender> userBlenders;
	private final long seed;
	private final double relevanceThreshold;
	private final int at;
	
	public MixedHybridRecommender(DataModel dataModel, ArrayList<RecommenderBuilder> builders, long seed, double relevanceThreshold, int at) throws TasteException {
		super(dataModel);
		this.builders = builders;
		this.recs = new ArrayList<Recommender>(builders.size());
		for (RecommenderBuilder builder : builders) {
			recs.add(builder.buildRecommender(dataModel));
		}
		this.nrecs = builders.size();
		this.userBlenders = new FastByIDMap<UserBlender>();
		this.seed = seed;
		this.relevanceThreshold = relevanceThreshold;
		this.at = at;
		trainBlenders();
	}
	
	public MixedHybridRecommender(DataModel dataModel, ArrayList<RecommenderBuilder> builders, long seed, double relevanceThreshold, int at, CandidateItemsStrategy strategy) throws TasteException {
		super(dataModel, strategy);
		this.builders = builders;
		this.recs = new ArrayList<Recommender>(builders.size());
		for (RecommenderBuilder builder : builders) {
			recs.add(builder.buildRecommender(dataModel, strategy));
		}
		this.nrecs = builders.size();
		this.userBlenders = new FastByIDMap<UserBlender>();
		this.seed = seed;
		this.relevanceThreshold = relevanceThreshold;
		this.at = at;
		trainBlenders();
	}
	
	private void trainBlenders() throws TasteException {
		
		KFoldDataSplitter folds = new KFoldDataSplitter(this.getDataModel(), 5, new Random(this.seed));
		Iterator<Fold> itF = folds.getFolds();
		while (itF.hasNext()) {

			Fold fold = itF.next();

			DataModel trainingModel = fold.getTraining();
			FastByIDMap<PreferenceArray> testPrefs = fold.getTesting();
			LongPrimitiveIterator it = fold.getUserIDs().iterator();

			List<RecWrapper> theRecs = new ArrayList<RecWrapper>(this.builders.size());
			for (RecommenderBuilder recommenderBuilder : this.builders) {
				theRecs.add(new RecWrapper(recommenderBuilder.buildRecommender(trainingModel, fold), 1.0, ""));
			}
			MetaRecommender theRecommender = new MetaRecommender(trainingModel, theRecs, this.candidateItemsStrategy);
			
			while (it.hasNext()) {

				long userID = it.nextLong();
				PreferenceArray prefs = testPrefs.get(userID);
				if (prefs == null || prefs.length() == 0) {
					continue; // Oops we excluded all prefs for the user -- just move on
				}
				
				UserBlender blender = this.userBlenders.get(userID);
				if (blender == null) {
					blender = new UserBlender(this.nrecs);
					this.userBlenders.put(userID, blender);	
				}
				
				try {
					theRecommender.getCandidateItems(userID);
				} catch (NoSuchUserException nsue) {
					continue;
				}

				FastIDSet relevantItemIDs = new FastIDSet(prefs.length());
				for (int i = 0; i < prefs.length(); i++) {
					if (prefs.getValue(i) >= this.relevanceThreshold) {
						relevantItemIDs.add(prefs.getItemID(i));
					}
				}

				int numRelevantItems = relevantItemIDs.size();
				if (numRelevantItems <= 0) {
					continue;
				}

				try {
					trainingModel.getPreferencesFromUser(userID);
				} catch (NoSuchUserException nsee) {
					continue; // Oops we excluded all prefs for the user -- just move on
				}

				List<List<RecommendedItem>> recommendedLists = theRecommender.recommendSeperately(userID, this.at, null, false);
				int index = 0;
				for (List<RecommendedItem> recommendedList : recommendedLists) {
					for (RecommendedItem recommendedItem : recommendedList) {
						if (relevantItemIDs.contains(recommendedItem.getItemID())) {
							blender.add(index);
						}
					}
					index++;
				}
			}
		}
	}

	@Override
	public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems)
			throws TasteException {
		
		List<RecommendedItem> recommendations = new ArrayList<RecommendedItem>();
		List<Long> ids = new ArrayList<Long>();
		
		boolean uniform = false;
		UserBlender blender = this.userBlenders.get(userID);
		if (blender == null || blender.getCount() == 0) {
			// No hit for all algorithms in validation set, so uniform combination
			uniform = true;
		}
		
		List<Integer> howManies = new ArrayList<Integer>(this.nrecs);
		int idMax = 0, max = -1, sum = 0;
		for (int idx = 0; idx < this.nrecs; idx++) {
			int howRealMany = 0;
			if (uniform) {
				log.warn("No blender for user {}, using uniform combination", userID);
				howRealMany = (int) ((float) howMany / (float) this.nrecs);
			} else {
				howRealMany = (int) ((float) blender.getNbHits(idx) / (float) blender.getCount() * (float) howMany);
			}
			howManies.add(howRealMany);
			if (howRealMany > max) {
				max = howRealMany;
				idMax = idx;
			}
			sum += howRealMany;
		}
		if (sum < howMany) {
			howManies.set(idMax, max + howMany - sum);
		}
//		log.info("User {} : {} items from recs", userID, howManies);
		
		int idx = 0;
		for (Recommender rec : this.recs) {
			List<RecommendedItem> l = rec.recommend(userID, howMany, rescorer, includeKnownItems);
			int k = 0;
			for (RecommendedItem item : l) {
				if (k >= howManies.get(idx)) {
					break;
				} else {
					if (!ids.contains(item.getItemID())) {
						recommendations.add(item);
						ids.add(item.getItemID());
					}
					k++;
				}
			}
			idx++;
		}
		return recommendations;
	}

	@Override
	public float estimatePreference(long userID, long itemID) throws TasteException {
		return Float.NaN;
	}

	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
	}

}
