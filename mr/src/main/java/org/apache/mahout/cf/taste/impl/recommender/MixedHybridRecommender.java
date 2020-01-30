package org.apache.mahout.cf.taste.impl.recommender;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.ChronologicalDataSplitter;
import org.apache.mahout.cf.taste.eval.ChronologicalPerUserDataSplitter;
import org.apache.mahout.cf.taste.eval.FoldDataSplitter;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.eval.Fold;
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
	private final int nbFolds;
	
	public MixedHybridRecommender(DataModel dataModel, ArrayList<RecommenderBuilder> builders, long seed, double relevanceThreshold, int at, int nbFolds) throws TasteException {
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
		trainBlenders();
	}
	
	public MixedHybridRecommender(DataModel dataModel, ArrayList<RecommenderBuilder> builders, long seed, double relevanceThreshold, int at, int nbFolds, CandidateItemsStrategy strategy) throws TasteException {
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
				
				// Start to describe opt pb
				Model model = new Model();
				BoolVar[][] cutoffs = model.boolVarMatrix("cutoffs", this.nrecs, this.at + 1);
				IntVar[] allHits = model.intVarArray("allHits", this.nrecs, 0, this.at);
				IntVar[] allRanks = model.intVarArray("allRanks", this.nrecs, 0, this.at);
				
				int[] ranks = new int[this.at + 1];
			    for (int rank = 0; rank <= this.at; rank++) {
			    	ranks[rank] = rank;
			    }
				
				// Set total number of recs
			    for (int index = 0; index < this.nrecs; index++) {
			    	model.scalar(cutoffs[index], ranks, "=", allRanks[index]).post();
			    }
			    model.sum(allRanks, "=", this.at).post();
				

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
				
				FastIDSet seen = new FastIDSet();
				
				for (List<RecommendedItem> recommendedList : recommendedLists) {
					
					int[] hits = new int[this.at + 1];
					
					// Constraint of only one cutoff to choose per algo
					model.sum(cutoffs[index], "=", 1).post();
					
					int thisIntersection = 0;
					int rank = 0;
					hits[rank] = thisIntersection;
					rank++;
					for (RecommendedItem recommendedItem : recommendedList) {
						long itemID = recommendedItem.getItemID();
						if (relevantItemIDs.contains(itemID) && !seen.contains(itemID)) {
							thisIntersection++;
						}
						seen.add(itemID);
						hits[rank] = thisIntersection;
						rank++;
					}
					
					// Fill to at with last value if not enough values (less recs than asked for)
					if (rank != this.at + 1) {
						for (int k = rank; k <= this.at; k++) {
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
				    	for (rank = 0; rank <= this.at; rank++) {
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
		
		int idx = 0;
		
//		return this.recs.get(idMax).recommend(userID, howMany, rescorer, includeKnownItems);
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
						k++;
					}
				}
			}
			idx++;
		}
		assert(recommendations.size() <= howMany);
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
