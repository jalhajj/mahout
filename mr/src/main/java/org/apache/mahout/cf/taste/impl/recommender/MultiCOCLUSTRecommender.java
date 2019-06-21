package org.apache.mahout.cf.taste.impl.recommender;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.eval.Fold;
import org.apache.mahout.cf.taste.impl.eval.KFoldRecommenderPredictionEvaluator;
import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class MultiCOCLUSTRecommender extends AbstractRecommender {

	
	private final COCLUSTRecommender r1;
	private final COCLUSTRecommender r2;
	private final COCLUSTRecommender r3;
	private static final Logger log = LoggerFactory.getLogger(MultiCOCLUSTRecommender.class);

	public MultiCOCLUSTRecommender(DataModel dataModel, int maxIter, CandidateItemsStrategy strategy)
			throws TasteException {
		super(dataModel, strategy);
		this.r1 = new COCLUSTRecommender(dataModel, 3, 3, maxIter, strategy);
		this.r2 = new COCLUSTRecommender(dataModel, 10, 10, maxIter, strategy);
		this.r3 = new COCLUSTRecommender(dataModel, 20, 20, maxIter, strategy);
	}

	public MultiCOCLUSTRecommender(DataModel dataModel, int maxIter, CandidateItemsStrategy strategy, double lambda)
			throws TasteException {
		super(dataModel, strategy);
		this.r1 = new COCLUSTRecommender(dataModel, 3, 3, maxIter, strategy, lambda);
		this.r2 = new COCLUSTRecommender(dataModel, 10, 10, maxIter, strategy, lambda);
		this.r3 = new COCLUSTRecommender(dataModel, 20, 20, maxIter, strategy, lambda);
	}

	public MultiCOCLUSTRecommender(DataModel dataModel, int maxIter) throws TasteException {
		super(dataModel);
		this.r1 = new COCLUSTRecommender(dataModel, 3, 3, maxIter);
		this.r2 = new COCLUSTRecommender(dataModel, 10, 10, maxIter);
		this.r3 = new COCLUSTRecommender(dataModel, 20, 20, maxIter);
	}

	@Override
	public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems)
			throws TasteException {
		Preconditions.checkArgument(howMany >= 0, "howMany must be at least 0");
		log.debug("Recommending items for user ID '{}'", userID);

		if (howMany == 0) {
			return Collections.emptyList();
		}

		PreferenceArray preferencesFromUser = getDataModel().getPreferencesFromUser(userID);
		FastIDSet possibleItemIDs = getAllOtherItems(userID, preferencesFromUser, includeKnownItems);

		List<RecommendedItem> topItems = TopItems.getTopItems(howMany, possibleItemIDs.iterator(), rescorer,
				new Estimator(userID));
		log.debug("Recommendations are: {}", topItems);

		return topItems;
	}

	/**
	 * a preference is estimated by considering the chessboard biclustering computed
	 */
	@Override
	public float estimatePreference(long userID, long itemID) throws TasteException {
		DataModel model = getDataModel();
		Float actualPref = model.getPreferenceValue(userID, itemID);
		if (actualPref != null) {
			return actualPref;
		}
		Float x1 = this.r1.estimatePreference(userID, itemID);
		Float x2 = this.r2.estimatePreference(userID, itemID);
		Float x3 = this.r3.estimatePreference(userID, itemID);
		
		double w1 = 3;
		double w2 = 2;
		double w3 = 1;
		
		return (float) ((x1 * w1 + x2 * w2 + x3 * w3) / (w1 + w2 + w3));

	}

	private final class Estimator implements TopItems.Estimator<Long> {

		private final long theUserID;

		private Estimator(long theUserID) {
			this.theUserID = theUserID;
		}

		@Override
		public double estimate(Long itemID) throws TasteException {
			return estimatePreference(theUserID, itemID);
		}
	}

	/**
	 * Refresh the data model and factorization.
	 */
	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
	}

}
