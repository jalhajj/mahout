package org.apache.mahout.cf.taste.impl.recommender;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Bicluster;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
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
			List<Bicluster<Long>> lowers = getLowers(sb);
			log.debug("Lower biclusters of {} are {}", sb, lowers);
			List<Bicluster<Long>> uppers = getUppers(sb);
			log.debug("Upper biclusters of {} are {}", sb, uppers);
			List<Bicluster<Long>> siblings = new ArrayList<Bicluster<Long>>();
			for (Bicluster<Long> b : uppers) {
				siblings.addAll(getLowers(b));
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

	private Set<Long> getSharedUsers(Set<Long> items) throws TasteException {
		DataModel model = getDataModel();
		Set<Long> shared = new HashSet<Long>();
		boolean first = true;
		for (long itemID : items) {
			if (first) {
				PreferenceArray prefs = model.getPreferencesForItem(itemID);
				for (Preference pref : prefs) {
					if (pref.getValue() >= this.threshold) {
						shared.add(pref.getUserID());
					}
				}
				first = false;
			} else {
				Set<Long> users = new HashSet<Long>(shared);
				for (long userID : users) {
					Float rating = model.getPreferenceValue(userID, itemID);
					if (rating == null || rating < this.threshold) {
						shared.remove(userID);
					}
				}
			}
		}
		return shared;
	}
	
	private Set<Long> getSharedItems(Set<Long> users) throws TasteException {
		DataModel model = getDataModel();
		Set<Long> shared = new HashSet<Long>();
		boolean first = true;
		for (long userID : users) {
			if (first) {
				PreferenceArray prefs = model.getPreferencesFromUser(userID);
				for (Preference pref : prefs) {
					if (pref.getValue() >= this.threshold) {
						shared.add(pref.getItemID());
					}
				}
				first = false;
			} else {
				Set<Long> items = new HashSet<Long>(shared);
				for (long itemID : items) {
					Float rating = model.getPreferenceValue(userID, itemID);
					if (rating == null || rating < this.threshold) {
						shared.remove(itemID);
					}
				}
			}
		}
		return shared;
	}
	
	private List<Bicluster<Long>> getLowers(Bicluster<Long> b) throws TasteException {
		DataModel model = getDataModel();
		Set<Long> items = b.getSetItems();
		List<Bicluster<Long>> lowers = new ArrayList<Bicluster<Long>>();
		LongPrimitiveIterator it = model.getItemIDs();
		while (it.hasNext()) {
			long itemID = it.nextLong();
			if (!items.contains(itemID)) {
				Set<Long> otherItems = new HashSet<Long>(items);
				otherItems.add(itemID);
				Set<Long> theUsers = getSharedUsers(otherItems);
				Set<Long> theItems = getSharedItems(theUsers);
				Bicluster<Long> theB = new Bicluster<Long>(theUsers, theItems);
				if (!theB.isEmpty()) {
					boolean valid = true;
					Bicluster<Long> toRemove = null;
					for (Bicluster<Long> bb : lowers) {
						if (!valid) {
							break;
						}
						if (bb.includeLower(theB)) {
							valid = false;
						} else if (bb.includeGreater(theB)) {
							toRemove = bb;
						}
					}
					if (valid ) {
						lowers.add(theB);
						if (toRemove != null) {
							lowers.remove(toRemove);
						}
					}
				}
			}
		}
		return lowers;
	}
	
	private List<Bicluster<Long>> getUppers(Bicluster<Long> b) throws TasteException {
		DataModel model = getDataModel();
		Set<Long> users = b.getSetUsers();
		List<Bicluster<Long>> uppers = new ArrayList<Bicluster<Long>>();
		LongPrimitiveIterator it = model.getUserIDs();
		while (it.hasNext()) {
			long userID = it.nextLong();
			if (!users.contains(userID)) {
				Set<Long> otherUsers = new HashSet<Long>(users);
				otherUsers.add(userID);
				Set<Long> theItems = getSharedItems(otherUsers);
				Set<Long> theUsers = getSharedUsers(theItems);
				Bicluster<Long> theB = new Bicluster<Long>(theUsers, theItems);
				if (!theB.isEmpty()) {
					boolean valid = true;
					Bicluster<Long> toRemove = null;
					for (Bicluster<Long> bb : uppers) {
						if (!valid) {
							break;
						}
						if (bb.includeGreater(theB)) {
							valid = false;
						} else if (bb.includeLower(theB)) {
							toRemove = bb;
						}
					}
					if (valid) {
						uppers.add(theB);
						if (toRemove != null) {
							uppers.remove(toRemove);
						}
					}
				}
			}
		}
		return uppers;
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
