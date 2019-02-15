package org.apache.mahout.cf.taste.impl.eval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.DataModelBuilder;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

/**
 * <p>
 * For each user, these implementation determine the top {@code n} preferences,
 * then evaluate the IR statistics based on a {@link DataModel} that does not
 * have these values. This number {@code n} is the "at" value, as in "precision
 * at 5". For example, this would mean precision evaluated by removing the top 5
 * preferences for a user and then finding the percentage of those 5 items
 * included in the top 5 recommendations for that user.
 * </p>
 */
public final class KFoldRecommenderIRStatsEvaluator implements RecommenderIRStatsEvaluator {
	
	private int noFolds;

	private static final Logger log = LoggerFactory.getLogger(KFoldRecommenderIRStatsEvaluator.class);

	private static final double LOG2 = Math.log(2.0);

	public static final double CHOOSE_THRESHOLD = Double.NaN;

	private final Random random;

	public KFoldRecommenderIRStatsEvaluator(int noFolds) {
		this.random = RandomUtils.getRandom();
		this.noFolds = noFolds;
	}

	@Override
	public IRStatistics evaluate(RecommenderBuilder recommenderBuilder, DataModelBuilder dataModelBuilder,
			DataModel dataModel, IDRescorer rescorer, int at, double relevanceThreshold, double evaluationPercentage)
			throws TasteException {

		Preconditions.checkArgument(recommenderBuilder != null, "recommenderBuilder is null");
		Preconditions.checkArgument(dataModel != null, "dataModel is null");
		Preconditions.checkArgument(at >= 1, "at must be at least 1");
		Preconditions.checkArgument(evaluationPercentage > 0.0 && evaluationPercentage <= 1.0,
		        "Invalid evaluationPercentage: " + evaluationPercentage + ". Must be: 0.0 < evaluationPercentage <= 1.0");

		log.info("Beginning evaluation using {} of {}", evaluationPercentage, dataModel);

		int numUsers = dataModel.getNumUsers();
		int numItems = dataModel.getNumItems();

		// Initialize buckets for the number of folds
		List<FastByIDMap<PreferenceArray>> folds = new ArrayList<FastByIDMap<PreferenceArray>>();
		for (int i = 0; i < noFolds; i++) {
			folds.add(new FastByIDMap<PreferenceArray>(1 + (int) (i / noFolds * numUsers)));
		}
		
		// Split the dataModel into K folds per user
	    LongPrimitiveIterator it = dataModel.getUserIDs();
	    while (it.hasNext()) {
	        long userID = it.nextLong();
	        if (random.nextDouble() < evaluationPercentage) {
	            splitOneUsersPrefs2(noFolds, folds, userID, dataModel);
	        }
	    }

		RunningAverage precision = new FullRunningAverage();
		RunningAverage recall = new FullRunningAverage();
		RunningAverage fallOut = new FullRunningAverage();
		RunningAverage nDCG = new FullRunningAverage();
		RunningAverage reach = new FullRunningAverage();
		
		// Rotate the folds. Each time only one is used for testing and the rest
	    // k-1 folds are used for training
	    for (int k = 0; k < noFolds; k++) {
	        FastByIDMap<PreferenceArray> trainingPrefs = new FastByIDMap<PreferenceArray>(
	                1 + (int) (evaluationPercentage * numUsers));
	        FastByIDMap<PreferenceArray> testPrefs = new FastByIDMap<PreferenceArray>(
	                1 + (int) (evaluationPercentage * numUsers));

	        for (int i = 0; i < folds.size(); i++) {

	            // The testing fold
	            testPrefs = folds.get(k);

	            // Build the training set from the remaining folds
	            if (i != k) {
	                for (Map.Entry<Long, PreferenceArray> entry : folds.get(i)
	                        .entrySet()) {
	                    if (!trainingPrefs.containsKey(entry.getKey())) {
	                        trainingPrefs.put(entry.getKey(), entry.getValue());
	                    } else {
	                        List<Preference> userPreferences = new ArrayList<Preference>();
	                        PreferenceArray existingPrefs = trainingPrefs
	                                .get(entry.getKey());
	                        for (int j = 0; j < existingPrefs.length(); j++) {
	                            userPreferences.add(existingPrefs.get(j));
	                        }

	                        PreferenceArray newPrefs = entry.getValue();
	                        for (int j = 0; j < newPrefs.length(); j++) {
	                            userPreferences.add(newPrefs.get(j));
	                        }
	                        trainingPrefs.remove(entry.getKey());
	                        trainingPrefs.put(entry.getKey(),
	                                new GenericUserPreferenceArray(
	                                        userPreferences));

	                    }
	                }
	            }
	        }

	        DataModel trainingModel = dataModelBuilder == null ? new GenericDataModel(
	                trainingPrefs) : dataModelBuilder
	                .buildDataModel(trainingPrefs);

	        Recommender recommender = recommenderBuilder
	                .buildRecommender(trainingModel);
	        
	        RunningAverage precisionFold = new FullRunningAverage();
			RunningAverage recallFold = new FullRunningAverage();
			RunningAverage fallOutFold = new FullRunningAverage();
			RunningAverage nDCGFold = new FullRunningAverage();
	        int numUsersRecommendedFor = 0;
			int numUsersWithRecommendations = 0;
	        
	        it = dataModel.getUserIDs();
	        while (it.hasNext()) {
	        	
	        	long userID = it.nextLong();
	        	
	        	PreferenceArray prefs = testPrefs.get(userID);
	        	if (prefs == null) {
	        		continue; // Oops we excluded all prefs for the user -- just move on
	        	}
	        	// List some most-preferred items that would count as (most) "relevant" results
				double theRelevanceThreshold = Double.isNaN(relevanceThreshold) ? computeThreshold(prefs)
						: relevanceThreshold;
				FastIDSet relevantItemIDs = new FastIDSet(at);
			    prefs.sortByValueReversed();
			    for (int i = 0; i < prefs.length() && relevantItemIDs.size() < at; i++) {
			      if (prefs.getValue(i) >= theRelevanceThreshold) {
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
				
				int size = numRelevantItems + trainingModel.getItemIDsFromUser(userID).size();
				if (size < 2 * at) {
					// Really not enough prefs to meaningfully evaluate this user
					continue;
				}
				
				int numRecommendedItems = 0;
				int intersectionSize = 0;
				List<RecommendedItem> recommendedItems = recommender.recommend(userID, at, rescorer);
				for (RecommendedItem recommendedItem : recommendedItems) {
					if (dataModel.getPreferenceValue(userID, recommendedItem.getItemID()) != null) {
						if (relevantItemIDs.contains(recommendedItem.getItemID())) {
							intersectionSize++;
						}
						numRecommendedItems++;
					}
				}

				// Precision
				if (numRecommendedItems > 0) {
					precisionFold.addDatum((double) intersectionSize / (double) numRecommendedItems);
				}

				// Recall
				recallFold.addDatum((double) intersectionSize / (double) numRelevantItems);

				// Fall-out
				if (numRelevantItems < size) {
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
			recall.addDatum(recallFold.getAverage());
			fallOut.addDatum(fallOutFold.getAverage());
			nDCG.addDatum(nDCGFold.getAverage());
			reach.addDatum((double) numUsersWithRecommendations / (double) numUsersRecommendedFor);
			
			log.info("Precision/recall/fall-out/nDCG/reach from fold {}: {} / {} / {} / {} / {}", k, precisionFold.getAverage(),
					recallFold.getAverage(), fallOutFold.getAverage(), nDCGFold.getAverage(),
					(double) numUsersWithRecommendations / (double) numUsersRecommendedFor);

	    }
	    
	    log.info("Precision/recall/fall-out/nDCG/reach: {} / {} / {} / {} / {}", precision.getAverage(),
				recall.getAverage(), fallOut.getAverage(), nDCG.getAverage(), reach.getAverage());

		return new IRStatisticsImpl(precision.getAverage(), recall.getAverage(), fallOut.getAverage(),
				nDCG.getAverage(), reach.getAverage());
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
	
	/**
	 * Split the preference values for one user into K folds, by shuffling.
	 * First Shuffle the Preference array for the user. Then distribute the item-preference pairs
	 * starting from the first buckets to the k-th bucket, and then start from the beggining.
	 * 
	 * @param k
	 * @param folds
	 * @param userID
	 * @param dataModel
	 * @throws TasteException
	 */
	private void splitOneUsersPrefs2(int k, List<FastByIDMap<PreferenceArray>> folds, long userID, DataModel dataModel) throws TasteException {

	    List<List<Preference>> oneUserPrefs = Lists.newArrayListWithCapacity(k + 1);
	    for (int i = 0; i < k; i++) {
	        oneUserPrefs.add(null);
	    }

	    PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
	    int size = prefs.length();


	    List<Preference> userPrefs = new ArrayList<>();
	    Iterator<Preference> it = prefs.iterator();
	    while (it.hasNext()) {
	        userPrefs.add(it.next());
	    }

	    // Shuffle the items
	    Collections.shuffle(userPrefs);

	    int currentBucket = 0;
	    for (int i = 0; i < size; i++) {
	        if (currentBucket == k) {
	            currentBucket = 0;
	        }

	        Preference newPref = new GenericPreference(userID, userPrefs.get(i).getItemID(), userPrefs.get(i).getValue());

	        if (oneUserPrefs.get(currentBucket) == null) {
	            oneUserPrefs.set(currentBucket, new ArrayList<Preference>());
	        }
	        oneUserPrefs.get(currentBucket).add(newPref);
	        currentBucket++;
	    }

	    for (int i = 0; i < k; i++) {
	        if (oneUserPrefs.get(i) != null) {
	            folds.get(i).put(userID, new GenericUserPreferenceArray(oneUserPrefs.get(i)));
	        }
	    }

	}

}
