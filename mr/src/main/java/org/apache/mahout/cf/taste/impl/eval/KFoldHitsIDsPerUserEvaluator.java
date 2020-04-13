package org.apache.mahout.cf.taste.impl.eval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.StringJoiner;

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

public final class KFoldHitsIDsPerUserEvaluator {

	private static final Logger log = LoggerFactory.getLogger(KFoldHitsIDsPerUserEvaluator.class);

	private final DataModel dataModel;
	private final FoldDataSplitter folds;

	public KFoldHitsIDsPerUserEvaluator(DataModel dataModel, int nbFolds, Random random) throws TasteException {
		Preconditions.checkArgument(dataModel != null, "dataModel is null");
		Preconditions.checkArgument(nbFolds > 1, "nbFolds must be > 1");

		this.dataModel = dataModel;
		this.folds = new ChronologicalPerUserDataSplitter(this.dataModel, (double) nbFolds / 100.0);
	}

	public KFoldHitsIDsPerUserEvaluator(DataModel dataModel, FoldDataSplitter splitter) throws TasteException {
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
		FastByIDMap<String> ids = new FastByIDMap<String>(n);
		
		Iterator<Fold> itF = this.folds.getFolds();
		int foldID = 0;
		while (itF.hasNext()) {
			
			log.debug("Fold #{}", foldID);

			Fold fold = itF.next();

			DataModel trainingModel = fold.getTraining();
			FastByIDMap<PreferenceArray> testPrefs = fold.getTesting();
			LongPrimitiveIterator it = fold.getUserIDs().iterator();

			Recommender recommender = recommenderBuilder.buildRecommender(trainingModel, fold);

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

				List<RecommendedItem> recommendedList = recommender.recommend(userID, at, null, false);
				
				
				
				FastIDSet recommended = new FastIDSet();
				for (RecommendedItem recommendedItem : recommendedList) {
					long itemID = recommendedItem.getItemID();
					recommended.add(itemID);
				}
				
				StringJoiner hits = new StringJoiner("|");
				StringJoiner misses = new StringJoiner("|");
				for (Long itemID : relevantItemIDs) {
					if (recommended.contains(itemID)) {
						hits.add(itemID.toString());
					} else {
						misses.add(itemID.toString());
					}
				}
				
				ids.put(userID, String.format("%s,%s", hits.toString(), misses.toString()));	

			}
			
			foldID++;

		}

		PerUserStatisticsImpl results = new PerUserStatisticsImpl(n);
		LongPrimitiveIterator it = ids.keySetIterator();
		while (it.hasNext()) {
			long userID = it.nextLong();
			results.addMisc(userID, ids.get(userID));
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
