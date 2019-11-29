package org.apache.mahout.cf.taste.impl.recommender;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;

public class MetaRecommender extends AbstractRecommender {
	
	public static class RecWrapper {
		
		private final Recommender rec;
		private double weight;
		
		public RecWrapper(Recommender rec, double weight) {
			this.rec = rec;
			this.weight = weight;
		}
		
		public Recommender getRecommender() {
			return this.rec;
		}
		
		public double getWeight() {
			return this.weight;
		}
		
		public void setWeight(double weight) {
			this.weight = weight;
		}
		
	}
	
	private final List<RecWrapper> recs;

	public MetaRecommender(DataModel dataModel, List<RecWrapper> recs) {
		super(dataModel);
		this.recs = recs;

	}
	
	public MetaRecommender(DataModel dataModel, List<RecWrapper> recs, CandidateItemsStrategy strategy) {
		super(dataModel, strategy);
		this.recs = recs;
	}
	
	@Override
	public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems)
			throws TasteException {
		List<RecommendedItem> recommendations = new ArrayList<RecommendedItem>();
		List<Long> ids = new ArrayList<Long>();
		for (RecWrapper rw : this.recs) {
			List<RecommendedItem> l = rw.getRecommender().recommend(userID, howMany, rescorer, includeKnownItems);
			int cnt = (int) (rw.getWeight() * (float) howMany);
			int k = 0;
			for (RecommendedItem item : l) {
				if (k >= cnt) {
					break;
				} else {
					if (!ids.contains(item.getItemID())) {
						recommendations.add(item);
						ids.add(item.getItemID());
					}
					k++;
				}
			}
		}
		return recommendations;
	}

	@Override
	public float estimatePreference(long userID, long itemID) throws TasteException {
		float sum = 0;
		int cnt = 0;
		for (RecWrapper rw : this.recs) {
			float x = rw.getRecommender().estimatePreference(userID, itemID);
			if (!Float.isNaN(x)) {
				sum += x;
				cnt++;
			}
		}
		if (cnt > 0) {
			return sum / (float) cnt;
		} else {
			return Float.NaN;
		}
	}

	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
	}

}
