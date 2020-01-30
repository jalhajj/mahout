package org.apache.mahout.cf.taste.impl.recommender;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class IdealMixedRecommender extends AbstractRecommender {
	
	private static final Logger log = LoggerFactory.getLogger(IdealMixedRecommender.class);
	
	private final ArrayList<RecommenderBuilder> builders;
	private final ArrayList<Recommender> recs;

	public IdealMixedRecommender(DataModel dataModel, ArrayList<RecommenderBuilder> builders) throws TasteException {
		super(dataModel);
		this.builders = builders;
		this.recs = new ArrayList<Recommender>(builders.size());
		for (RecommenderBuilder builder : builders) {
			recs.add(builder.buildRecommender(dataModel));
		}
	}
	
	public IdealMixedRecommender(DataModel dataModel, ArrayList<RecommenderBuilder> builders, CandidateItemsStrategy strategy) throws TasteException {
		super(dataModel, strategy);
		this.builders = builders;
		this.recs = new ArrayList<Recommender>(builders.size());
		for (RecommenderBuilder builder : builders) {
			recs.add(builder.buildRecommender(dataModel, strategy));
		}
	}
	
	public int getNbRecs() {
		return this.recs.size();
	}
	
	@Override
	public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems)
			throws TasteException {
		List<RecommendedItem> recommendations = new ArrayList<RecommendedItem>();
		List<Long> ids = new ArrayList<Long>();
		for (Recommender rec : this.recs) {
			List<RecommendedItem> l = rec.recommend(userID, howMany, rescorer, includeKnownItems);
			for (RecommendedItem item : l) {
				if (!ids.contains(item.getItemID())) {
					recommendations.add(item);
					ids.add(item.getItemID());
				}
			}
		}
		return recommendations;
	}
	
	public List<List<RecommendedItem>> recommendSeperately(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems)
			throws TasteException {
		List<List<RecommendedItem>> recommendations = new ArrayList<List<RecommendedItem>>();
		for (Recommender rec : this.recs) {
			List<RecommendedItem> l = rec.recommend(userID, howMany, rescorer, includeKnownItems);
			recommendations.add(l);
		}
		return recommendations;
	}
	
	public String getRecNames() {
		StringBuilder builder = new StringBuilder();
		for (Recommender rec : this.recs) {
			builder.append(rec.getClass().getSimpleName());
			builder.append(",");
		}
		return builder.toString();
	}

	@Override
	public float estimatePreference(long userID, long itemID) throws TasteException {
		return Float.NaN;
	}

	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
	}

}
