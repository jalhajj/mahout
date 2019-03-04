package org.apache.mahout.cf.taste.impl.eval;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

public final class KFoldRecommenderIRStatsEvaluator {

	private static final Logger log = LoggerFactory.getLogger(KFoldRecommenderIRStatsEvaluator.class);

	private static final double LOG2 = Math.log(2.0);

	private final DataModel dataModel;
	private final FoldDataSplitter folds;

	public KFoldRecommenderIRStatsEvaluator(DataModel dataModel, int nbFolds) throws TasteException {
		Preconditions.checkArgument(dataModel != null, "dataModel is null");
		Preconditions.checkArgument(nbFolds > 1, "nbFolds must be > 1");
		
		this.dataModel = dataModel;
		this.folds = new FoldDataSplitter(dataModel, nbFolds);
	}

	public IRStatistics evaluate(RecommenderBuilder recommenderBuilder, int at, double relevanceThreshold)
			throws TasteException {

		Preconditions.checkArgument(recommenderBuilder != null, "recommenderBuilder is null");
		Preconditions.checkArgument(at >= 1, "at must be at least 1");
		log.info("Beginning evaluation");

		int numItems = dataModel.getNumItems();

		RunningAverage precision = new FullRunningAverage();
		RunningAverage adjPrecision = new FullRunningAverage();
		RunningAverage recall = new FullRunningAverage();
		RunningAverage adjRecall = new FullRunningAverage();
		RunningAverage fallOut = new FullRunningAverage();
		RunningAverage nDCG = new FullRunningAverage();
		RunningAverage reach = new FullRunningAverage();

		Iterator<Fold> itF = this.folds.getFolds();
		int k = 0;
		while (itF.hasNext()) {
			
			Fold fold = itF.next();
			
			DataModel trainingModel = fold.getTraining();
			FastByIDMap<PreferenceArray> testPrefs = fold.getTesting();

			Recommender recommender = recommenderBuilder.buildRecommender(trainingModel);
			
			RunningAverage precisionFold = new FullRunningAverage();
			RunningAverage recallFold = new FullRunningAverage();
			RunningAverage adjPrecisionFold = new FullRunningAverage();
			RunningAverage adjRecallFold = new FullRunningAverage();
			RunningAverage fallOutFold = new FullRunningAverage();
			RunningAverage nDCGFold = new FullRunningAverage();
			int numUsersRecommendedFor = 0;
			int numUsersWithRecommendations = 0;

			LongPrimitiveIterator it = dataModel.getUserIDs();
			while (it.hasNext()) {

				long userID = it.nextLong();

				PreferenceArray prefs = testPrefs.get(userID);
				if (prefs == null) {
					log.debug("Ignoring user {}", userID);
					continue; // Oops we excluded all prefs for the user -- just move on
				}
				// List some most-preferred items that would count as (most) "relevant" results
				double theRelevanceThreshold = Double.isNaN(relevanceThreshold) ? computeThreshold(prefs)
						: relevanceThreshold;
				FastIDSet relevantItemIDs = new FastIDSet(prefs.length());
				FastIDSet notRelevantItemIDs = new FastIDSet(prefs.length());
				int adjNumRelevantItems = 0;
				for (int i = 0; i < prefs.length(); i++) {
					if (prefs.getValue(i) >= theRelevanceThreshold) {
						relevantItemIDs.add(prefs.getItemID(i));
						try {
							dataModel.getPreferencesForItem(prefs.getItemID(i));
							adjNumRelevantItems++;
						} catch (NoSuchItemException nsie) {
							// The item is not in training set, it will never be recommended, so do not
							// count it for adjusted recall
						}
					} else {
						notRelevantItemIDs.add(prefs.getItemID(i));
					}
				}

				int numRelevantItems = relevantItemIDs.size();
				int numNotRelevantItems = notRelevantItemIDs.size();
				if (numRelevantItems <= 0 || numNotRelevantItems <= 0) {
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
				int adjNumRecommendedItems = 0;
				int intersectionSize = 0;
				List<RecommendedItem> recommendedItems = recommender.recommend(userID, at);

				log.debug("Recommended items are {}", recommendedItems);
				List<Float> relevantScores = new ArrayList<Float>(relevantItemIDs.size());
				List<Float> trueScores = new ArrayList<Float>(relevantItemIDs.size());
				for (long itemID : relevantItemIDs) {
					relevantScores.add(recommender.estimatePreference(userID, itemID));
					trueScores.add(dataModel.getPreferenceValue(userID, itemID));

				}
				log.debug("Relevant items are {}, their predicted scores are {}, true scores are {}", relevantItemIDs,
						relevantScores, trueScores);

				for (RecommendedItem recommendedItem : recommendedItems) {

					if (relevantItemIDs.contains(recommendedItem.getItemID())) {
						intersectionSize++;
						adjNumRecommendedItems++;
					} else if (notRelevantItemIDs.contains(recommendedItem.getItemID())) {
						adjNumRecommendedItems++;
					}
					numRecommendedItems++;
				}

				log.debug("User {}: #rec {} / #relevant {} / #goodrec {}", userID, numRecommendedItems,
						numRelevantItems, intersectionSize);

				// Precision
				if (numRecommendedItems > 0) {
					precisionFold.addDatum((double) intersectionSize / (double) numRecommendedItems);
				}
				if (adjNumRecommendedItems > 0) {
					adjPrecisionFold.addDatum((double) intersectionSize / (double) adjNumRecommendedItems);
				}

				// Recall
				if (numRelevantItems > 0) {
					recallFold.addDatum((double) intersectionSize / (double) numRelevantItems);
				}
				if (adjNumRelevantItems > 0) {
					adjRecallFold.addDatum((double) intersectionSize / (double) adjNumRelevantItems);
				}

				// Fall-out
				if (numRelevantItems < prefs.length()) {
					fallOutFold.addDatum(
							(double) (numRecommendedItems - intersectionSize) / (double) (numItems - numRelevantItems));
				}

				// nDCG
				// In computing, assume relevant IDs have relevance 1 and others 0
				double cumulativeGain = 0.0;
				double idealizedGain = 0.0;
				for (int i = 0; i < numRecommendedItems; i++) {
					RecommendedItem item = recommendedItems.get(i);
					double discount = 1.0 / log2(i + 2.0); // Classical formulation says log(i+1), but i is 0-based here
					if (relevantItemIDs.contains(item.getItemID())) {
						cumulativeGain += discount;
					}
					// otherwise we're multiplying discount by relevance 0 so it doesn't do anything

					// Ideally results would be ordered with all relevant ones first, so this
					// theoretical
					// ideal list starts with number of relevant items equal to the total number of
					// relevant items
					if (i < numRelevantItems) {
						idealizedGain += discount;
					}
				}
				if (idealizedGain > 0.0) {
					nDCGFold.addDatum(cumulativeGain / idealizedGain);
				}

				// Reach
				numUsersRecommendedFor++;
				if (numRecommendedItems > 0) {
					numUsersWithRecommendations++;
				}

			}

			precision.addDatum(precisionFold.getAverage());
			adjPrecision.addDatum(adjPrecisionFold.getAverage());
			recall.addDatum(recallFold.getAverage());
			adjRecall.addDatum(adjRecallFold.getAverage());
			fallOut.addDatum(fallOutFold.getAverage());
			nDCG.addDatum(nDCGFold.getAverage());
			reach.addDatum((double) numUsersWithRecommendations / (double) numUsersRecommendedFor);

			log.info(
					"Precision/recall/fall-out/nDCG/reach/adjusted precision/adjusted recall from fold {}: {} / {} / {} / {} / {} / {} / {}",
					k++, precisionFold.getAverage(), recallFold.getAverage(), fallOutFold.getAverage(),
					nDCGFold.getAverage(), (double) numUsersWithRecommendations / (double) numUsersRecommendedFor,
					adjPrecisionFold.getAverage(), adjRecallFold.getAverage());

		}

		log.info(
				"Precision/recall/fall-out/nDCG/reach/adjusted precision/adjusted recall: {} / {} / {} / {} / {} / {} / {}",
				precision.getAverage(), recall.getAverage(), fallOut.getAverage(), nDCG.getAverage(),
				reach.getAverage(), adjPrecision.getAverage(), adjRecall.getAverage());

		return new IRStatisticsImpl(precision.getAverage(), recall.getAverage(), fallOut.getAverage(),
				nDCG.getAverage(), reach.getAverage(), adjPrecision.getAverage(), adjRecall.getAverage());
	}

	private static double computeThreshold(PreferenceArray prefs) {
		if (prefs.length() < 2) {
			// Not enough data points -- return a threshold that allows everything
			return Double.NEGATIVE_INFINITY;
		}
		RunningAverageAndStdDev stdDev = new FullRunningAverageAndStdDev();
		int size = prefs.length();
		for (int i = 0; i < size; i++) {
			stdDev.addDatum(prefs.getValue(i));
		}
		return stdDev.getAverage() + stdDev.getStandardDeviation();
	}

	private static double log2(double value) {
		return Math.log(value) / LOG2;
	}

}
