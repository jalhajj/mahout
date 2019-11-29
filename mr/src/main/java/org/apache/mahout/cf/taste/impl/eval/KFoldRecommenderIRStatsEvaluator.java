package org.apache.mahout.cf.taste.impl.eval;

import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.FoldDataSplitter;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.similarity.UncenteredCosineSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

public final class KFoldRecommenderIRStatsEvaluator {

	private static final Logger log = LoggerFactory.getLogger(KFoldRecommenderIRStatsEvaluator.class);

	private static final double LOG2 = Math.log(2.0);

	private final DataModel dataModel;
	private final FoldDataSplitter folds;

	public KFoldRecommenderIRStatsEvaluator(DataModel dataModel, int nbFolds, Random random) throws TasteException {
		Preconditions.checkArgument(dataModel != null, "dataModel is null");
		Preconditions.checkArgument(nbFolds > 1, "nbFolds must be > 1");

		this.dataModel = dataModel;
		this.folds = new KFoldDataSplitter(dataModel, nbFolds, random);
	}

	public KFoldRecommenderIRStatsEvaluator(DataModel dataModel, FoldDataSplitter splitter) throws TasteException {
		Preconditions.checkArgument(dataModel != null, "dataModel is null");
		Preconditions.checkArgument(splitter != null, "splitter is null");

		this.dataModel = dataModel;
		this.folds = splitter;
	}

	public void restrainUserIDsWithCoverage(RecommenderBuilder recommenderBuilder, int at) throws TasteException {
		Preconditions.checkArgument(recommenderBuilder != null, "recommenderBuilder is null");
		Iterator<Fold> itF = this.folds.getFolds();
		while (itF.hasNext()) {
			Fold fold = itF.next();
			DataModel trainingModel = fold.getTraining();
			LongPrimitiveIterator it = fold.getUserIDs().iterator();
			Recommender recommender = recommenderBuilder.buildRecommender(trainingModel, fold);
			FastIDSet ids = new FastIDSet();
			while (it.hasNext()) {
				long userID = it.nextLong();
				List<RecommendedItem> recommendedItems = recommender.recommend(userID, at);
				int numRecommendedItems = recommendedItems.size();
				if (numRecommendedItems < 1) {
					ids.add(userID);
				}
			}
			fold.removeUserIDs(ids);
		}
	}

	public IRStatistics evaluate(RecommenderBuilder recommenderBuilder, int at, double relevanceThreshold)
			throws TasteException {

		Preconditions.checkArgument(recommenderBuilder != null, "recommenderBuilder is null");
		Preconditions.checkArgument(at >= 1, "at must be at least 1");
		log.info("Beginning evaluation");

		int numItems = dataModel.getNumItems();

		ItemSimilarity similarity = new UncenteredCosineSimilarity(this.dataModel);

		RunningAverage precision = new FullRunningAverage();
		RunningAverage recall = new FullRunningAverage();
		RunningAverage fallOut = new FullRunningAverage();
		RunningAverage nDCG = new FullRunningAverage();
		RunningAverage reachAtLeastOne = new FullRunningAverage();
		RunningAverage reachAll = new FullRunningAverage();
		RunningAverage itemCoverage = new FullRunningAverage();
		RunningAverage perPrecision = new FullRunningAverage();
		RunningAverage perRecall = new FullRunningAverage();
		RunningAverage diversity = new FullRunningAverage();
		
		int listsCnt = 0;
		int samePredictionCnt = 0;

		Iterator<Fold> itF = this.folds.getFolds();
		int k = 0;
		while (itF.hasNext()) {

			Fold fold = itF.next();

			DataModel trainingModel = fold.getTraining();
			FastByIDMap<PreferenceArray> testPrefs = fold.getTesting();
			LongPrimitiveIterator it = fold.getUserIDs().iterator();

			Recommender recommender = recommenderBuilder.buildRecommender(trainingModel, fold);

			RunningAverage precisionFold = new FullRunningAverage();
			RunningAverage recallFold = new FullRunningAverage();
			RunningAverage fallOutFold = new FullRunningAverage();
			RunningAverage nDCGFold = new FullRunningAverage();
			RunningAverage perPrecisionFold = new FullRunningAverage();
			RunningAverage perRecallFold = new FullRunningAverage();
			RunningAverage diversityFold = new FullRunningAverage();
			int numUsersRecommendedFor = 0;
			int numUsersWithRecommendations = 0;
			int numUsersWithAllRecommendations = 0;

			FastIDSet recItems = new FastIDSet();

			while (it.hasNext()) {

				long userID = it.nextLong();

				FastIDSet candidateItemsIDs;
				try {
					candidateItemsIDs = recommender.getCandidateItems(userID);
				} catch (NoSuchUserException nsue) {
					continue;
				}
				int numCandidateItems = candidateItemsIDs.size();

				PreferenceArray prefs = testPrefs.get(userID);
				if (prefs == null) {
					log.debug("Ignoring user {}", userID);
					continue; // Oops we excluded all prefs for the user -- just move on
				}
				// List some most-preferred items that would count as (most) "relevant" results
				double theRelevanceThreshold = Double.isNaN(relevanceThreshold)
						? computeThreshold(trainingModel.getPreferencesFromUser(userID))
						: relevanceThreshold;
				FastIDSet relevantItemIDs = new FastIDSet(prefs.length());
				for (int i = 0; i < prefs.length(); i++) {
					if (prefs.getValue(i) >= theRelevanceThreshold) {
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
				List<RecommendedItem> recommendedItems = recommender.recommend(userID, at);
				for (RecommendedItem recommendedItem : recommendedItems) {
//					log.warn("For user {}: {},{}", userID, recommendedItem.getItemID(), recommendedItem.getValue());
					recItems.add(recommendedItem.getItemID());
					if (relevantItemIDs.contains(recommendedItem.getItemID())) {
						intersectionSize++;
					}
					numRecommendedItems++;
				}

				log.debug("User {}: #rec {} / #relevant {} / #goodrec {}", userID, numRecommendedItems,
						numRelevantItems, intersectionSize);

				// Precision
				double p = 0;
				if (numRecommendedItems > 0) {
					p = (double) intersectionSize / (double) at;
					p = p > 1 ? 1 : p;
					precisionFold.addDatum(p);
				}

				// Per Precision
				double maxPrecision = Math.min(1, (double) numRelevantItems / (double) numRecommendedItems);
				if (maxPrecision > 0) {
					perPrecisionFold.addDatum(p / maxPrecision);
				}

				// Recall
				double r = 0;
				if (numRelevantItems > 0) {
					r = (double) intersectionSize / (double) numRelevantItems;
					recallFold.addDatum(r);
				}

				// Per Recall
				double maxRecall = Math.min(1, (double) numRecommendedItems / (double) numRelevantItems);
				if (maxRecall > 0) {
					perRecallFold.addDatum(r / maxRecall);
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
				if (numRecommendedItems >= at) {
					numUsersWithAllRecommendations++;
				}

				// Diversity
				RunningAverage diversityUser = new FullRunningAverage();
				for (RecommendedItem recommendedItem : recommendedItems) {
					for (RecommendedItem otherRecommendedItem : recommendedItems) {
						if (recommendedItem != otherRecommendedItem) {
							Double x = similarity.itemSimilarity(recommendedItem.getItemID(),
									otherRecommendedItem.getItemID());
							if (!x.isNaN()) {
								diversityUser.addDatum(x);
							}
						}
					}
				}
				if (diversityUser.getCount() > 0) {
					diversityFold.addDatum(1 - diversityUser.getAverage());
				}
				
				// Same prediction for all the list
				Float first = Float.NaN;
				boolean same = true;
				for (RecommendedItem recommendedItem : recommendedItems) {
					if (first.isNaN()) {
						first = recommendedItem.getValue();
					} else {
						if (recommendedItem.getValue() != first) {
							same = false;
							break;
						}
					}
				}
				if (same) {
					samePredictionCnt++;
				}
				listsCnt++;

			}

			precision.addDatum(precisionFold.getAverage());
			recall.addDatum(recallFold.getAverage());
			fallOut.addDatum(fallOutFold.getAverage());
			nDCG.addDatum(nDCGFold.getAverage());
			perPrecision.addDatum(perPrecisionFold.getAverage());
			perRecall.addDatum(perRecallFold.getAverage());
			reachAtLeastOne.addDatum((double) numUsersWithRecommendations / (double) numUsersRecommendedFor);
			reachAll.addDatum((double) numUsersWithAllRecommendations / (double) numUsersRecommendedFor);
			itemCoverage.addDatum((double) recItems.size() / (double) numItems);
			diversity.addDatum(diversityFold.getAverage());

			log.info(
					"Precision/recall/fall-out/nDCG/reachAtLeastOne/reachAll/itemCoverage/diversity from fold {}: {} / {} / {} / {} / {} / {} / {} / {}",
					k++, precisionFold.getAverage(), recallFold.getAverage(), fallOutFold.getAverage(),
					nDCGFold.getAverage(), (double) numUsersWithRecommendations / (double) numUsersRecommendedFor,
					(double) numUsersWithAllRecommendations / (double) numUsersRecommendedFor,
					(double) recItems.size() / (double) numItems, diversityFold.getAverage());

		}

		log.info(
				"Precision/recall/fall-out/nDCG/reachAtLeastOne/reachAll/itemCoverage/diversity: {} / {} / {} / {} / {} / {} / {} / {}",
				precision.getAverage(), recall.getAverage(), fallOut.getAverage(), nDCG.getAverage(),
				reachAtLeastOne.getAverage(), reachAtLeastOne.getAverage(), itemCoverage.getAverage(),
				diversity.getAverage());
		
		log.info("Proportion of lists with same prediction score for all recommendations: {}", (double) samePredictionCnt / (double) listsCnt);

		IRStatistics results = new IRStatisticsImpl(precision.getAverage(), recall.getAverage(), fallOut.getAverage(),
				nDCG.getAverage(), reachAtLeastOne.getAverage(), reachAll.getAverage(), itemCoverage.getAverage(),
				perPrecision.getAverage(), perRecall.getAverage(), diversity.getAverage());

		return results;

	}

	private static double computeThreshold(PreferenceArray prefs) {
		if (prefs.length() < 2) {
			// Not enough data points -- return a threshold that allows everything
			return Double.NEGATIVE_INFINITY;
		}
		RunningAverage avg = new FullRunningAverage();
		int size = prefs.length();
		for (int i = 0; i < size; i++) {
			avg.addDatum(prefs.getValue(i));
		}
		return avg.getAverage();
	}

	private static double log2(double value) {
		return Math.log(value) / LOG2;
	}

}
