package org.apache.mahout.cf.taste.impl.eval;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

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
		while (itF.hasNext()) {

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
				
				int index = 0;
				for (List<RecommendedItem> recommendedList : recommendedLists) {
					int thisIntersection = 0;
					for (RecommendedItem recommendedItem : recommendedList) {
						if (relevantItemIDs.contains(recommendedItem.getItemID())) {
							intersectionSize++;
							thisIntersection++;
						}
						numRecommendedItems++;
					}
					hitsFrom.get(userID).get(index).addDatum(thisIntersection);
					index++;
				}

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

}
