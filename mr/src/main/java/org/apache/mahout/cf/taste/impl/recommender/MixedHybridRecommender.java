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
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
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

import org.chocosolver.solver.Model;
import org.chocosolver.solver.variables.BoolVar;
import org.chocosolver.solver.variables.IntVar;
import org.chocosolver.util.ESat;

public class MixedHybridRecommender extends AbstractRecommender {
	
	private static final Logger log = LoggerFactory.getLogger(MixedHybridRecommender.class);
	
	private static final double LOG2 = Math.log(2.0);

	static class UserBlender {
		
		private final ArrayList<RunningAverage> gains;
		
		UserBlender(int nbAlgos) {
			this.gains = new ArrayList<RunningAverage>(nbAlgos);
			for (int i = 0; i < nbAlgos; i++) {
				this.gains.add(new FullRunningAverage());
			}
		}
		
		double getTotal() {
			double totalGain = 0.0;
			for (RunningAverage avg : this.gains) {
				totalGain += avg.getAverage();
			}
			return totalGain;
		}
		
		double get(int idx) {
			return this.gains.get(idx).getAverage();
		}
		
		void add(int idx, double value) {
			this.gains.get(idx).addDatum(value);
		}
		
		public String toString() {
			return gains.toString();
		}
		
	}
	
	private final ArrayList<RecommenderBuilder> builders;
	private final ArrayList<Recommender> recs;
	private final int nrecs;
	private final FastByIDMap<UserBlender> userBlenders;
	private final long seed;
	private final double relevanceThreshold;
	private final int at;
	
	private final Random rand;
	private ArrayList<Integer> stats;
	
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
		this.rand = new Random(seed);
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
		this.rand = new Random(seed);
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
				
				Model model = new Model();
				BoolVar[][] cutoffs = model.boolVarMatrix("cutoffs", this.nrecs, this.at);
				IntVar[] allHits = model.intVarArray("allHits", this.nrecs, 0, this.at);

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
					
					int[] hits = new int[this.at];
					
					// Constraint of only one cutoff to choose per algo
					model.sum(cutoffs[index], "=", 1).post();
					
					int thisIntersection = 0;
					int rank = 0;
					for (RecommendedItem recommendedItem : recommendedList) {
						if (relevantItemIDs.contains(recommendedItem.getItemID())) {
							thisIntersection++;
						}
						
						hits[rank] = thisIntersection;
						rank++;
					}
					
					// Fill to at with last value if not enough values (less recs than asked for)
					if (rank != this.at) {
						for (int k = rank; k < this.at; k++) {
							hits[k] = thisIntersection;
						}
					}
					
					// Set hit value for this algo in function of the cutoff variables
					model.scalar(cutoffs[index], hits, "=", allHits[index]).post();
					
					
					index++;
				}
				
				// Objective function
				IntVar OBJ = model.intVar("objective", 0, this.nrecs * this.at);
				model.sum(allHits, "=", OBJ).post();
				model.setObjective(Model.MAXIMIZE, OBJ);
				
				// Solve
				if(model.getSolver().solve()) {
					
				    for (index = 0; index < this.nrecs; index++) {
				    	int rank;
				    	for (rank = 0; rank < this.at; rank++) {
				    		if (cutoffs[index][rank].getBooleanValue() == ESat.eval(true)) {
				    			break;
				    		}
				    	}
				    	double prop = (double) rank / (double) this.at;
				    	blender.add(index, prop);
				    }
				    
				} else {
					log.warn("LP could not be solved");
					for (index = 0; index < this.nrecs; index++) {
						blender.add(index, 0);
				    }
				}
				
			}
		}
	}

	@Override
	public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems)
			throws TasteException {
		
		if (userID == 1) {
			this.stats = new ArrayList<Integer>(Collections.nCopies(this.nrecs, 0));
		}
		
		List<RecommendedItem> recommendations = new ArrayList<RecommendedItem>();
		List<Long> ids = new ArrayList<Long>();
		
		boolean defaultBlender = false;
		UserBlender blender = this.userBlenders.get(userID);
		if (blender == null || blender.getTotal() <= 0.0) {
			// No hit for all algorithms in validation set, so uniform combination
			defaultBlender = true;
		}
		
		List<Integer> howManies = new ArrayList<Integer>(this.nrecs);
		int idMax = 0, max = -1, sum = 0;
		for (int idx = 0; idx < this.nrecs; idx++) {
			int howRealMany = 0;
			if (defaultBlender) {
				if (idx == 0) {
					howRealMany = howMany;
				} else {
					howRealMany = 0;
				}
//				log.warn("No blender for user {}, using default combination", userID);
//				howRealMany = (int) ((float) howMany / (float) this.nrecs);
			} else {
				howRealMany = (int) ((float) blender.get(idx) * (float) howMany);
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
		
//		log.info("For user {}: {}", userID, howManies);
		
//		this.stats.set(idMax, this.stats.get(idMax) + 1);
//		if (userID == 943) {
//			log.info("Stats : {}", this.stats);
//		}
		
		int idx = 0;
//		return this.recs.get(idMax).recommend(userID, howMany, rescorer, includeKnownItems);
		for (Recommender rec : this.recs) {
			
			int start = 0;//Math.max(0, Math.min(howMany - howManies.get(idx), blender.getInfRank(idx)));
			
			List<RecommendedItem> l = rec.recommend(userID, howMany, rescorer, includeKnownItems);
			
//			Collections.shuffle(l, this.rand);
			
			int k = 0;
			for (RecommendedItem item : l) {
				if (k < start) {
					continue;
				} else if (k >= howManies.get(idx) + start) {
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
	
	private static double log2(double value) {
		return Math.log(value) / LOG2;
	}

}
