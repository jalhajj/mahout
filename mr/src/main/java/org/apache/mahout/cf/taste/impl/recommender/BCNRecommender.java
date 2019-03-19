package org.apache.mahout.cf.taste.impl.recommender;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Bicluster;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.impl.similarity.JaccardItemSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class BCNRecommender extends AbstractRecommender {

	private final ItemSimilarity sim;
	private final float threshold;
	private final FastByIDMap<Bicluster<Long>> smallers;
	private final FastByIDMap<List<Bicluster<Long>>> neighborhoods;

	private static final Logger log = LoggerFactory.getLogger(BCNRecommender.class);

	public BCNRecommender(DataModel dataModel, float threshold, CandidateItemsStrategy strategy) throws TasteException {
		super(dataModel, strategy);
		this.sim = new JaccardItemSimilarity(dataModel, threshold);
		this.threshold = threshold;
		this.smallers = new FastByIDMap<Bicluster<Long>>(dataModel.getNumUsers());
		this.neighborhoods = new FastByIDMap<List<Bicluster<Long>>>(dataModel.getNumUsers());
	}

	public BCNRecommender(DataModel dataModel, float threshold) throws TasteException {
		super(dataModel);
		this.sim = new JaccardItemSimilarity(dataModel, threshold);
		this.threshold = threshold;
		this.smallers = new FastByIDMap<Bicluster<Long>>(dataModel.getNumUsers());
		this.neighborhoods = new FastByIDMap<List<Bicluster<Long>>>(dataModel.getNumUsers());
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

	@Override
	public float estimatePreference(long userID, long itemID) throws TasteException {
		DataModel model = getDataModel();
		Float actualPref = model.getPreferenceValue(userID, itemID);
		if (actualPref != null) {
			return actualPref;
		}

		Bicluster<Long> sb = getSmallestBicluster(userID);
		if (sb.isEmpty()) {
			return Float.NaN;
		}
		
		double g = 0;
		int cnt = 0;
		Iterator<Long> it = sb.getItems();
		while (it.hasNext()) {
			long j = it.next();
			if (j != itemID) {
				g += this.sim.itemSimilarity(itemID, j);
				cnt++;
			}
		}
		if (cnt > 0) {
			g = g / (double) cnt;
		}
		log.debug("Global similarity of user {} and item {} is {}", userID, itemID, g);
		
		List<Bicluster<Long>> neighbors = getBiclusterNeighborhood(sb, userID);
		double l = 0;
		for (Bicluster<Long> bb : neighbors) {
			if (bb.containsItem(itemID)) {
				l += biSim(sb, bb);
			}
		}
		if (l == 0) {
			return Float.NaN;
		}
		log.debug("Local similarity of user {} and item {} is {}", userID, itemID, l);

		return (float) (g * l);
	}

	private Bicluster<Long> getSmallestBicluster(long userID) throws TasteException {
		
		if (!this.smallers.containsKey(userID)) {
		
			DataModel model = getDataModel();
			Bicluster<Long> sb = new Bicluster<Long>();
			sb.addUser(userID);
			long itemID = 0;
			for (Preference pref : model.getPreferencesFromUser(userID)) {
				if (pref.getValue() >= this.threshold) {
					sb.addItem(pref.getItemID());
					itemID = pref.getItemID();
				}
			}
			if (!sb.isEmpty()) {
				for (Preference pref : model.getPreferencesForItem(itemID)) {
					long otherUserID = pref.getUserID();
					boolean hasAll = true;
					Iterator<Long> it = sb.getItems();
					while (hasAll && it.hasNext()) {
						long otherItemID = it.next();
						Float rating = model.getPreferenceValue(otherUserID, otherItemID);
						if (rating == null || rating < this.threshold) {
							hasAll = false;
						}
					}
					if (hasAll) {
						sb.addUser(otherUserID);
					}
				}
			}
			this.smallers.put(userID, sb);
			log.debug("Computed smallest bicluster for user {} is {}", userID, sb);
			return sb;
		
		} else {
			Bicluster<Long> sb = this.smallers.get(userID);
			log.debug("Cached smallest bicluster for user {} is {}", userID, sb);
			return sb;
		}
	}
	
	private List<Bicluster<Long>> getBiclusterNeighborhood(Bicluster<Long> sb, long userID) throws TasteException {
		if (!this.neighborhoods.containsKey(userID)) {
			List<Bicluster<Long>> lowers = new ArrayList<Bicluster<Long>>();
			getLowers(sb, lowers);
			log.debug("Lower biclusters of {} are {}", sb, lowers);
			List<Bicluster<Long>> uppers = new ArrayList<Bicluster<Long>>();
			getUppers(sb, uppers);
			log.debug("Upper biclusters of {} are {}", sb, uppers);
			List<Bicluster<Long>> siblings = new ArrayList<Bicluster<Long>>();
			for (Bicluster<Long> b : uppers) {
				getLowers(b, siblings);
			}
			log.debug("Sibling biclusters of {} are {}", sb, siblings);
			
			List<Bicluster<Long>> candidates = new ArrayList<Bicluster<Long>>();
			candidates.addAll(lowers);
			for (Bicluster<Long> b : siblings) {
				if (!b.equals(sb)) {
					candidates.add(b);
				}
			}
			log.debug("Computed candidate biclusters of {} are {}", sb, candidates);
			
			this.neighborhoods.put(userID, candidates);
			return candidates;
		} else {
			List<Bicluster<Long>> candidates = this.neighborhoods.get(userID);
			log.debug("Cached candidate biclusters of {} are {}", sb, candidates);
			return candidates;
		}
	}
	
	private void getLowers(Bicluster<Long> b, List<Bicluster<Long>> biclusters) throws TasteException {
		long[] userIDs = new long[b.getNbUsers()];
		Iterator<Long> it = b.getUsers();
		int i = 0;
		while (it.hasNext()) {
			long userID = it.next();
			userIDs[i] = userID;
			i++;
		}
		getLowers(b, biclusters, userIDs, i - 1);
	}
	
	private void getLowers(Bicluster<Long> b, List<Bicluster<Long>> biclusters, long[] userIDs, int i) throws TasteException {
		DataModel model = getDataModel();
		if (!b.isEmpty()) {
			int nbItemsAdded = 0;
			Iterator<Long> it = b.getUsers();
			long userID = it.next();
			for (Preference pref : model.getPreferencesFromUser(userID)) {
				long itemID = pref.getItemID();
				boolean hasAll = true;
				it = b.getUsers();
				while (hasAll && it.hasNext()) {
					long otherUserID = it.next();
					Float rating = model.getPreferenceValue(otherUserID, itemID);
					if (rating == null || rating < this.threshold) {
						hasAll = false;
					}
				}
				if (hasAll && !b.containsItem(itemID)) {
					b.addItem(itemID);
					nbItemsAdded++;
				}
			}
			if (nbItemsAdded > 0) {
				boolean valid = true;
				for (Bicluster<Long> bb : biclusters) {
					if (!valid) {
						break;
					}
					if (bb.includeLower(b)) {
						valid = false;
					}
				}
				if (valid) {
					biclusters.add(b);
				}
			} else {
				if (i >= 0) {
					long otherUserID = userIDs[i];
					getLowers(b.copy(), biclusters, userIDs, i - 1);
					Bicluster<Long> bWithout = b.copy();
					bWithout.removeUser(otherUserID);
					getLowers(bWithout, biclusters, userIDs, i - 1);
				} 
			}
		}
	}
	
	private void getUppers(Bicluster<Long> b, List<Bicluster<Long>> biclusters) throws TasteException {
		long[] itemIDs = new long[b.getNbItems()];
		Iterator<Long> it = b.getItems();
		int i = 0;
		while (it.hasNext()) {
			long itemID = it.next();
			itemIDs[i] = itemID;
			i++;
		}
		getUppers(b, biclusters, itemIDs, i - 1);
	}
	
	private void getUppers(Bicluster<Long> b, List<Bicluster<Long>> biclusters, long[] itemIDs, int i) throws TasteException {
		DataModel model = getDataModel();
		if (!b.isEmpty()) {
			int nbUsersAdded = 0;
			Iterator<Long> it = b.getItems();
			long itemID = it.next();
			for (Preference pref : model.getPreferencesForItem(itemID)) {
				long userID = pref.getUserID();
				boolean hasAll = true;
				it = b.getItems();
				while (hasAll && it.hasNext()) {
					long otherItemID = it.next();
					Float rating = model.getPreferenceValue(userID, otherItemID);
					if (rating == null || rating < this.threshold) {
						hasAll = false;
					}
				}
				if (hasAll && !b.containsUser(userID)) {
					b.addUser(userID);
					nbUsersAdded++;
				}
			}
			if (nbUsersAdded > 0) {
				boolean valid = true;
				for (Bicluster<Long> bb : biclusters) {
					if (!valid) {
						break;
					}
					if (bb.includeGreater(b)) {
						valid = false;
					}
				}
				if (valid) {
					biclusters.add(b);
				}
			} else {
				if (i >= 0) {
					long otherItemID = itemIDs[i];
					getUppers(b.copy(), biclusters, itemIDs, i - 1);
					Bicluster<Long> bWithout = b.copy();
					bWithout.removeItem(otherItemID);
					getUppers(bWithout, biclusters, itemIDs, i - 1);
				} 
			}
		}
	}
	
	private double biSim(Bicluster<Long> b1, Bicluster<Long> b2) throws TasteException {
		DataModel model = getDataModel();
		Bicluster<Long> b = b1.copy();
		b.merge(b2);
		int zeros = 0;
		Iterator<Long> itU = b.getUsers();
		while (itU.hasNext()) {
			long userID = itU.next();
			Iterator<Long> itI = b.getItems();
			while (itI.hasNext()) {
				long itemID = itI.next();
				Float rating = model.getPreferenceValue(userID, itemID);
				if (rating == null || rating < this.threshold) {
					zeros++;
				}
			}
		}
		int cnt = b.getNbUsers() * b.getNbItems();
		double x = (double) zeros / (double) cnt;
		return 1 - x;
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

	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
	}

}
