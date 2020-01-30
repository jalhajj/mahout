package org.apache.mahout.cf.taste.impl.eval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.ChronologicalDataSplitter;
import org.apache.mahout.cf.taste.eval.ChronologicalPerUserDataSplitter;
import org.apache.mahout.cf.taste.eval.FoldDataSplitter;
import org.apache.mahout.cf.taste.eval.PerUserStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.recommender.IdealMixedRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

import org.chocosolver.solver.Model;
import org.chocosolver.solver.variables.BoolVar;
import org.chocosolver.solver.variables.IntVar;
import org.chocosolver.util.ESat;

public final class KFoldIdealMixedRecommenderPerUserEvaluator {

	private static final Logger log = LoggerFactory.getLogger(KFoldIdealMixedRecommenderPerUserEvaluator.class);

	private final DataModel dataModel;
	private final FoldDataSplitter folds;

	public KFoldIdealMixedRecommenderPerUserEvaluator(DataModel dataModel, int nbFolds, Random random) throws TasteException {
		Preconditions.checkArgument(dataModel != null, "dataModel is null");
		Preconditions.checkArgument(nbFolds > 1, "nbFolds must be > 1");

		this.dataModel = dataModel;
		this.folds = new ChronologicalPerUserDataSplitter(this.dataModel, (double) nbFolds / 100.0);
	}

	public KFoldIdealMixedRecommenderPerUserEvaluator(DataModel dataModel, FoldDataSplitter splitter) throws TasteException {
		Preconditions.checkArgument(dataModel != null, "dataModel is null");
		Preconditions.checkArgument(splitter != null, "splitter is null");

		this.dataModel = dataModel;
		this.folds = splitter;
	}

	public PerUserStatistics evaluate(RecommenderBuilder recommenderBuilder, int at, Double relevanceThreshold) throws TasteException {

		Preconditions.checkArgument(recommenderBuilder != null, "recommenderBuilder is null");
		Preconditions.checkArgument(at >= 1, "at must be at least 1");
		Preconditions.checkArgument(!relevanceThreshold.isNaN(), "relevanceThreshold is NaN");
		log.info("Beginning evaluation");

		int n = this.dataModel.getNumUsers();
		FastByIDMap<RunningAverage> precision = new FastByIDMap<RunningAverage>(n);
		FastByIDMap<RunningAverage> optPrecision = new FastByIDMap<RunningAverage>(n);
		FastByIDMap<ArrayList<RunningAverage>> hitsFrom = new FastByIDMap<ArrayList<RunningAverage>>(n);
		
		Iterator<Fold> itF = this.folds.getFolds();
		int foldID = 0;
		while (itF.hasNext()) {
			
			log.debug("Fold #{}", foldID);

			Fold fold = itF.next();

			DataModel trainingModel = fold.getTraining();
			FastByIDMap<PreferenceArray> testPrefs = fold.getTesting();
			LongPrimitiveIterator it = fold.getUserIDs().iterator();

			Recommender theRecommender = recommenderBuilder.buildRecommender(trainingModel, fold);
			if (!(theRecommender instanceof IdealMixedRecommender)) {
				log.error("KFoldMetaRecommenderPerUserEvaluator: MetaRecommender required");
				return null;
			}
			IdealMixedRecommender recommender = (IdealMixedRecommender) theRecommender;

			while (it.hasNext()) {

				long userID = it.nextLong();
				
				PreferenceArray prefs = testPrefs.get(userID);
				if (prefs == null || prefs.length() == 0) {
					log.debug("Ignoring user {}", userID);
					continue; // Oops we excluded all prefs for the user -- just move on
				}
				
				try {
					recommender.getCandidateItems(userID);
				} catch (NoSuchUserException nsue) {
					continue;
				}
				
				int nrecs = recommender.getNbRecs();
				
				// Start to describe opt pb
				Model model = new Model();
				BoolVar[][] cutoffs = model.boolVarMatrix("cutoffs", nrecs, at + 1);
				IntVar[] allHits = model.intVarArray("allHits", nrecs, 0, at);
				IntVar[] allRanks = model.intVarArray("allRanks", nrecs, 0, at);
				
				int[] ranks = new int[at + 1];
			    for (int rank = 0; rank <= at; rank++) {
			    	ranks[rank] = rank;
			    }
				
				// Set total number of recs
			    for (int index = 0; index < nrecs; index++) {
			    	model.scalar(cutoffs[index], ranks, "=", allRanks[index]).post();
			    }
			    model.sum(allRanks, "=", at).post();

				FastIDSet relevantItemIDs = new FastIDSet(prefs.length());
				for (int i = 0; i < prefs.length(); i++) {
					if (prefs.getValue(i) >= relevanceThreshold) {
						relevantItemIDs.add(prefs.getItemID(i));
					}
				}

				int numRelevantItems = relevantItemIDs.size();
				if (numRelevantItems <= 0) {
					log.debug("Ignoring user {}", userID);
					continue;
				}

				try {
					trainingModel.getPreferencesFromUser(userID);
				} catch (NoSuchUserException nsee) {
					log.debug("Ignoring user {}", userID);
					continue; // Oops we excluded all prefs for the user -- just move on
				}

				int numRecommendedItems = 0;
				int intersectionSize = 0;
				List<List<RecommendedItem>> recommendedLists = recommender.recommendSeperately(userID, at, null, false);
				
				int nb = recommendedLists.size();
				if (!hitsFrom.containsKey(userID)) {
					ArrayList<RunningAverage> l = new ArrayList<RunningAverage>(nb);
					for (int k = 0; k < nb; k++) {
						l.add(new FullRunningAverage());
					}
					hitsFrom.put(userID, l);
				}
				
				FastByIDMap<Integer> occurences = new FastByIDMap<Integer>();
				Set<Long> hits = new HashSet<Long>();
				
				int index = 0;
				for (List<RecommendedItem> recommendedList : recommendedLists) {
				
					int[] phits = new int[at + 1];
					
					List<HitStats> stats = new ArrayList<HitStats>();
					
					int thisIntersection = 0;
					int rank = 0;
					phits[rank] = thisIntersection;
					rank++;
					for (RecommendedItem recommendedItem : recommendedList) {
						long itemID = recommendedItem.getItemID();
						if (relevantItemIDs.contains(itemID)) {
							intersectionSize++;
							thisIntersection++;
							stats.add(new HitStats(itemID, rank));
							hits.add(itemID);
						}
						
						if (!occurences.containsKey(itemID)) {
							occurences.put(itemID, 1);
						} else {
							occurences.put(itemID, occurences.get(itemID) + 1);
						}

						phits[rank] = thisIntersection;
						rank++;
						numRecommendedItems++;
					}
					hitsFrom.get(userID).get(index).addDatum(thisIntersection);
					
					// Fill to at with last value if not enough values (less recs than asked for)
					if (rank != at + 1) {
						for (int k = rank; k <= at; k++) {
							phits[k] = thisIntersection;
						}
					}
					
					// Set hit value for this algo in function of the cutoff variables
					model.scalar(cutoffs[index], phits, "=", allHits[index]).post();
					
					log.debug("Rec items for user {} from rec {}: {}", userID, index, stats);
					
					index++;
				}
				
				// Objective function
				IntVar OBJ = model.intVar("objective", 0, nrecs * at);
				model.sum(allHits, "=", OBJ).post();
				model.setObjective(Model.MAXIMIZE, OBJ);
				
				double optP = 0;
				
				// Solve
				if(model.getSolver().solve()) {
					
					FastIDSet selected = new FastIDSet(at);
					index = 0;
					for (List<RecommendedItem> recommendedList : recommendedLists) {
				    	int rank;
				    	for (rank = 0; rank <= at; rank++) {
				    		if (cutoffs[index][rank].getBooleanValue() == ESat.eval(true)) {
				    			break;
				    		}
				    	}
				    	index++;
				    	int k = 0;
				    	for (RecommendedItem recommendedItem : recommendedList) {
				    		if (k > rank) {
				    			break;
				    		}
				    		selected.add(recommendedItem.getItemID());
				    		k++;
				    	}
				    }
					
					int thisIntersection = 0;
					for (long itemID : selected) {
						if (relevantItemIDs.contains(itemID)) {
							thisIntersection++;
						}
					}
					optP = (double) thisIntersection / (double) at;
				    
				}
				
				log.debug("Hit items for user {}: {}", userID, hits);
				
//				List<HitStats> hstats = new ArrayList<HitStats>(occurences.size());
//				LongPrimitiveIterator hiterator = occurences.keySetIterator();
//				while (hiterator.hasNext()) {
//					long itemID = hiterator.nextLong();
//					int occ = occurences.get(itemID);
//					hstats.add(new HitStats(itemID, occ));
//				}
//				Collections.sort(hstats, new HitStatsComparator());
//				log.info("Sorted occ items for user {}: {}", userID, hstats);
//				log.info("");

				// Precision // FIXME computation is wrong, some hits are counted several times!!!
				double p = 0;
				if (numRecommendedItems > 0) {
					p = (double) intersectionSize / (double) at;
					p = p > 1 ? 1 : p;
					if (!precision.containsKey(userID)) {
						precision.put(userID, new FullRunningAverage());
					}
					precision.get(userID).addDatum(p);
					
					if (!optPrecision.containsKey(userID)) {
						optPrecision.put(userID, new FullRunningAverage());
					}
					optPrecision.get(userID).addDatum(optP);
				}

			}
			
			foldID++;

		}

		PerUserStatisticsImpl results = new PerUserStatisticsImpl(n);
		LongPrimitiveIterator it;
		
		it = precision.keySetIterator();
		while (it.hasNext()) {
			long userID = it.nextLong();
			results.addPrecision(userID, precision.get(userID).getAverage());
		}
		
		it = optPrecision.keySetIterator();
		while (it.hasNext()) {
			long userID = it.nextLong();
			results.addOther(userID, optPrecision.get(userID).getAverage());
		}
		
		it = hitsFrom.keySetIterator();
		while (it.hasNext()) {
			long userID = it.nextLong();
			List<Double> l = new ArrayList<Double>();
			for (RunningAverage avg : hitsFrom.get(userID)) {
				l.add(avg.getAverage());
			}
			results.addHitsFrom(userID, l);
		}
		
		return results;
	}
	
	class HitStats {
		
		private final long itemID;
		private final int count;
		
		HitStats(long itemID, int count) {
			this.itemID = itemID;
			this.count = count;
		}
		
		public String toString() {
			return String.format("%d (%d)", this.itemID, this.count);
		}
		
	}
	
	class HitStatsComparator implements Comparator<HitStats> {

		@Override
		public int compare(HitStats h1, HitStats h2) {
			int x = - Integer.compare(h1.count, h2.count);
			if (x == 0) {
				return Long.compare(h1.itemID, h2.itemID);
			} else {
				return x;
			}
		}
		
	}

}
