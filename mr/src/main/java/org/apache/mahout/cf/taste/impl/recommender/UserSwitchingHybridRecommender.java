package org.apache.mahout.cf.taste.impl.recommender;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.ChronologicalPerUserDataSplitter;
import org.apache.mahout.cf.taste.eval.FoldDataSplitter;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.eval.Fold;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UserSwitchingHybridRecommender extends AbstractRecommender {
	
	private static final Logger log = LoggerFactory.getLogger(UserSwitchingHybridRecommender.class);
	
	class BlenderStats {
		
		private int hits;
		private int bestRank;
		private final int idx;
		
		BlenderStats(int idx) {
			this.hits = 0;
			this.bestRank = Integer.MAX_VALUE;
			this.idx = idx;
		}
		
		void incrWithRank(int rank) {
			this.hits++;
			if (rank < this.bestRank) {
				this.bestRank = rank;
			}
		}
		
		public String toString() {
			return String.format("{ %d (%d) }", this.hits, this.bestRank);
		}
	}
	
	class BlenderStatsComparator implements Comparator<BlenderStats> {

		@Override
		public int compare(BlenderStats s1, BlenderStats s2) {
			int x = Integer.compare(s1.hits, s2.hits);
			if (x == 0) {
				x = -Integer.compare(s1.bestRank, s2.bestRank);
			}
			return -x;
		}
		
	}

	class UserBlender {
		
		private int DEFAULT_TO_SET = -47575;
		
		private final ArrayList<BlenderStats> stats;
		private Integer bestIndex;
		
		UserBlender(int nbAlgos) {
			this.stats = new ArrayList<BlenderStats>(nbAlgos);
			for (int i = 0; i < nbAlgos; i++) {
				this.stats.add(new BlenderStats(i));
			}
			bestIndex = null;
		}
		
		boolean checkIfBestEnough(int threshold) {
			int first = Integer.MIN_VALUE;
			int second = Integer.MIN_VALUE;
			for (BlenderStats e : this.stats) {
				int n = e.hits;
				if (n > first) {
					second = first;
					first = n;
				} else {
					if (n > second) {
						second = n;
					}
				}
			}
			return first >= (second + threshold);
		}
		
		int getAttribution() {
			try {
				return this.bestIndex;
			} catch (NullPointerException ex) {
				log.error("getAttribution: tried to get best index but training not done, returning 0");
				return 0;
			}
		}
		
		private int getBestIdx() {
			List<BlenderStats> toSort = new ArrayList<BlenderStats>(this.stats.size());
			for (BlenderStats e : this.stats) {
				toSort.add(e);
			}
			toSort.sort(new BlenderStatsComparator());
			return toSort.get(0).idx;
		}
		
		void incrWithRank(int idx, int rank) {
			this.stats.get(idx).incrWithRank(rank);
		}
		
		void doneTraining(AlgAttributionStats s, int threshold) {
			if (checkIfBestEnough(threshold)) {	
				this.bestIndex = getBestIdx();
				s.incr(this.bestIndex);
			} else {
				this.bestIndex = DEFAULT_TO_SET;
			}
		}
		
		void checkReallyDone(AlgAttributionStats s) {
			if (this.bestIndex == DEFAULT_TO_SET) {
				this.bestIndex = s.getBest();
			}
		}
		
		public String toString() {
			return this.stats.toString();
		}
		
	}
	
	class AlgAttributionStats {
		
		private final ArrayList<Integer> stats;
		
		AlgAttributionStats(int nbAlgos) {
			this.stats = new ArrayList<Integer>(nbAlgos);
			for (int i = 0; i < nbAlgos; i++) {
				this.stats.add(0);
			}
		}
		
		void incr(int idx) {
			this.stats.set(idx, this.stats.get(idx) + 1);
		}
		
		int getBest() {
			int max = Collections.max(this.stats);
			return this.stats.indexOf(max);
		}
		
		public String toString() {
			return this.stats.toString();
		}
		
	}
	
	private final ArrayList<RecommenderBuilder> builders;
	private final ArrayList<Recommender> recs;
	private final int nrecs;
	private final FastByIDMap<UserBlender> userBlenders;
	private final long seed;
	private final double relevanceThreshold;
	private final int at;
	private final int nbFolds;
	private final int bestThreshold;
	private final AlgAttributionStats algAttributionStats;
	
	public UserSwitchingHybridRecommender(DataModel dataModel, ArrayList<RecommenderBuilder> builders, long seed, double relevanceThreshold, int at, int nbFolds, int bestThreshold) throws TasteException {
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
		this.nbFolds = nbFolds;
		this.bestThreshold = bestThreshold;
		this.algAttributionStats = new AlgAttributionStats(this.nrecs);
		trainBlenders();
	}
	
	public UserSwitchingHybridRecommender(DataModel dataModel, ArrayList<RecommenderBuilder> builders, long seed, double relevanceThreshold, int at, int nbFolds, int bestThreshold, CandidateItemsStrategy strategy) throws TasteException {
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
		this.nbFolds = nbFolds;
		this.bestThreshold = bestThreshold;
		this.algAttributionStats = new AlgAttributionStats(this.nrecs);
		trainBlenders();
	}
	
	private void trainBlenders() throws TasteException {
		
		FoldDataSplitter folds = new ChronologicalPerUserDataSplitter(this.getDataModel(), (double) this.nbFolds / 100);
		Iterator<Fold> itF = folds.getFolds();
		while (itF.hasNext()) {

			Fold fold = itF.next();

			DataModel trainingModel = fold.getTraining();
			
			FastByIDMap<PreferenceArray> testPrefs = fold.getTesting();
			LongPrimitiveIterator it = fold.getUserIDs().iterator();

			IdealMixedRecommender theRecommender = new IdealMixedRecommender(trainingModel, this.builders, this.candidateItemsStrategy);
			
			while (it.hasNext()) {

				long userID = it.nextLong();
				PreferenceArray prefs = testPrefs.get(userID);
				if (prefs == null || prefs.length() == 0) {
					continue; // Oops we excluded all prefs for the user -- just move on
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
				
				UserBlender blender = this.userBlenders.get(userID);
				if (blender == null) {
					blender = new UserBlender(this.nrecs);
					this.userBlenders.put(userID, blender);	
				}

				List<List<RecommendedItem>> recommendedLists = theRecommender.recommendSeperately(userID, this.at, null, false);
				int index = 0;
				
				for (List<RecommendedItem> recommendedList : recommendedLists) {
					int rank = 0;
					rank++;
					for (RecommendedItem recommendedItem : recommendedList) {
						long itemID = recommendedItem.getItemID();
						if (relevantItemIDs.contains(itemID)) {
							blender.incrWithRank(index, rank);
						}
						rank++;
					}
					index++;
				}
				blender.doneTraining(this.algAttributionStats, this.bestThreshold);
			}
			
			it = fold.getUserIDs().iterator();
			while (it.hasNext()) {
				long userID = it.nextLong();				
				UserBlender blender = this.userBlenders.get(userID);
				if (blender != null) {	
					blender.checkReallyDone(this.algAttributionStats);
				}
			}
		}
		log.info("Attribution stats: {}", this.algAttributionStats);
	}

	@Override
	public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems)
			throws TasteException {

		boolean defaultBlender = false;
		UserBlender blender = this.userBlenders.get(userID);
		if (blender == null) {
			defaultBlender = true;
		}
		
		int idxRec;
		if (!defaultBlender) {
			idxRec = blender.getAttribution();
		} else {
			idxRec = this.algAttributionStats.getBest();
		}
		
		return this.recs.get(idxRec).recommend(userID, howMany, rescorer, includeKnownItems);

	}

	@Override
	public float estimatePreference(long userID, long itemID) throws TasteException {
		return Float.NaN;
	}

	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
	}

}
