package org.apache.mahout.cf.taste.impl.eval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.FoldDataSplitter;
import org.apache.mahout.cf.taste.eval.PerUserStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.recommender.MetaRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

public final class KFoldMetaRecommenderPerUserEvaluator {

	private static final Logger log = LoggerFactory.getLogger(KFoldMetaRecommenderPerUserEvaluator.class);

	private final DataModel dataModel;
	private final FoldDataSplitter folds;

	public KFoldMetaRecommenderPerUserEvaluator(DataModel dataModel, int nbFolds, Random random) throws TasteException {
		Preconditions.checkArgument(dataModel != null, "dataModel is null");
		Preconditions.checkArgument(nbFolds > 1, "nbFolds must be > 1");

		this.dataModel = dataModel;
		this.folds = new KFoldDataSplitter(this.dataModel, nbFolds, random);
	}

	public KFoldMetaRecommenderPerUserEvaluator(DataModel dataModel, FoldDataSplitter splitter) throws TasteException {
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
		FastByIDMap<ArrayList<RunningAverage>> hitsFrom = new FastByIDMap<ArrayList<RunningAverage>>(n);
		
		Iterator<Fold> itF = this.folds.getFolds();
		int foldID = 0;
		while (itF.hasNext()) {
			
			log.info("Fold #{}", foldID);

			Fold fold = itF.next();

			DataModel trainingModel = fold.getTraining();
			FastByIDMap<PreferenceArray> testPrefs = fold.getTesting();
			LongPrimitiveIterator it = fold.getUserIDs().iterator();

			Recommender theRecommender = recommenderBuilder.buildRecommender(trainingModel, fold);
			if (!(theRecommender instanceof MetaRecommender)) {
				log.error("KFoldMetaRecommenderPerUserEvaluator: MetaRecommender required");
				return null;
			}
			MetaRecommender recommender = (MetaRecommender) theRecommender;

			while (it.hasNext()) {

				long userID = it.nextLong();
				
				if (userID != 6) {
					continue;
				}
				
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
					
					List<HitStats> stats = new ArrayList<HitStats>();
					
					int thisIntersection = 0;
					int rank = 0;
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

						rank++;
						numRecommendedItems++;
					}
					hitsFrom.get(userID).get(index).addDatum(thisIntersection);
					
					log.info("Rec items for user {} from rec {}: {}", userID, index, stats);
					
					index++;
				}
				
				log.info("Hit items for user {}: {}", userID, hits);
				
				List<HitStats> hstats = new ArrayList<HitStats>(occurences.size());
				LongPrimitiveIterator hiterator = occurences.keySetIterator();
				while (hiterator.hasNext()) {
					long itemID = hiterator.nextLong();
					int occ = occurences.get(itemID);
					hstats.add(new HitStats(itemID, occ));
				}
				Collections.sort(hstats, new HitStatsComparator());
				log.info("Sorted occ items for user {}: {}", userID, hstats);
				log.info("");

				// Precision
				double p = 0;
				if (numRecommendedItems > 0) {
					p = (double) intersectionSize / (double) at;
					p = p > 1 ? 1 : p;
					if (!precision.containsKey(userID)) {
						precision.put(userID, new FullRunningAverage());
					}
					precision.get(userID).addDatum(p);
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
