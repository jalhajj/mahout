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
		private String name;
		
		public RecWrapper(Recommender rec, double weight, String name) {
			this.rec = rec;
			this.weight = weight;
			this.name = name;
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
		
		public String getName() {
			return this.name;
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
	
	public int getNbRecs() {
		return this.recs.size();
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
	
	public List<List<RecommendedItem>> recommendSeperately(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems)
			throws TasteException {
		List<List<RecommendedItem>> recommendations = new ArrayList<List<RecommendedItem>>();
		for (RecWrapper rw : this.recs) {
			int cnt = (int) (rw.getWeight() * (float) howMany);
			List<RecommendedItem> l = rw.getRecommender().recommend(userID, cnt, rescorer, includeKnownItems);
			recommendations.add(l);
		}
		return recommendations;
	}
	
	public String getRecNames() {
		StringBuilder builder = new StringBuilder();
		for (RecWrapper rw : this.recs) {
			builder.append(rw.getName());
			builder.append(",");
		}
		return builder.toString();
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
