package org.apache.mahout.cf.taste.impl.recommender;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Average;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class AdaptativeCOCLUSTRecommender extends AbstractRecommender {

	private final Random random;
	private COCLUSTRec curRec;
	private final int nbMaxIterations;

	private static final Logger log = LoggerFactory.getLogger(AdaptativeCOCLUSTRecommender.class);

	public AdaptativeCOCLUSTRecommender(DataModel dataModel, int maxIter, CandidateItemsStrategy strategy)
			throws TasteException {
		super(dataModel, strategy);
		this.random = RandomUtils.getRandom();
		this.nbMaxIterations = maxIter;
		init();
	}

	public AdaptativeCOCLUSTRecommender(DataModel dataModel, int maxIter) throws TasteException {
		super(dataModel);
		this.random = RandomUtils.getRandom();
		this.nbMaxIterations = maxIter;
		init();
	}

	private void init() throws TasteException {

		this.curRec = new COCLUSTRec(1, 1);
		TrainingError err = this.curRec.getTrainingError();
		for (int s = 0; s < this.nbMaxIterations; s++) {
			MaxPair max = err.getMaximum();
			double ref = err.getTotal();
			log.info("Current parameters are k={} and l={}, training error is {}", this.curRec.k, this.curRec.l, ref);

			int opt = random.nextInt(3);
			if (opt == 0) {
				this.curRec.splitUserCluster(max.getRow());
			} else if (opt == 1) {
				this.curRec.splitItemCluster(max.getCol());
			} else {
				this.curRec.splitUserCluster(max.getRow());
				this.curRec.splitItemCluster(max.getCol());
			}
			this.curRec.iterate(this.nbMaxIterations);
			TrainingError newErr = this.curRec.getTrainingError();
			double newRef = newErr.getTotal();
			if (newRef > ref) {
				this.curRec.revertClusterSplit();
				log.info("New error is {}, let's revert", newRef);
			} else {
				err = newErr;
				log.info("It's better, new parameters are {} and {}", this.curRec.k, this.curRec.l);
			}
		}
		this.curRec.iterate(this.nbMaxIterations);
		this.curRec.postProcess();
		log.info("Final parameters are k={} and l={}", this.curRec.k, this.curRec.l);
	}

	class COCLUSTRec {

		COCLUSTRec(int nbUserClusters, int nbItemClusters) throws TasteException {
			this.k = nbUserClusters;
			this.l = nbItemClusters;
			init();
		}

		private int k;
		private int l;
		private ArrayList<ArrayList<Average>> ACOC;
		private ArrayList<Average> ARC;
		private ArrayList<Average> ACC;
		private FastByIDMap<Average> AR;
		private FastByIDMap<Average> AC;
		private FastByIDMap<Index> Rho;
		private FastByIDMap<Index> Gamma;
		
		private FastByIDMap<Index> bakRho;
		private FastByIDMap<Index> bakGamma;
		private ArrayList<ArrayList<Average>> bakACOC;
		private ArrayList<Average> bakARC;
		private ArrayList<Average> bakACC;
		
		private FastByIDMap<Float> bias;

		private void init() throws TasteException {
			DataModel dataModel = getDataModel();
			int n = dataModel.getNumUsers();
			int m = dataModel.getNumItems();
			this.AR = new FastByIDMap<Average>(n);
			this.AC = new FastByIDMap<Average>(m);
			this.Rho = new FastByIDMap<Index>(n);
			this.Gamma = new FastByIDMap<Index>(m);
			this.bias = new FastByIDMap<Float>(n);
			train();
		}

		private void randomInit() throws TasteException {
			DataModel dataModel = getDataModel();
			LongPrimitiveIterator it;
			it = dataModel.getUserIDs();
			while (it.hasNext()) {
				this.Rho.put(it.nextLong(), new Index(random.nextInt(this.k)));
			}
			it = dataModel.getItemIDs();
			while (it.hasNext()) {
				this.Gamma.put(it.nextLong(), new Index(random.nextInt(this.l)));
			}
		}

		private void train() throws TasteException {

			/* Randomly initialize biclusters */
			randomInit();

			/* Pre-compute AR and AC */
			DataModel dataModel = getDataModel();
			LongPrimitiveIterator itU = dataModel.getUserIDs();
			while (itU.hasNext()) {
				long userID = itU.nextLong();
				PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
				for (Preference preference : prefs) {
					long itemID = preference.getItemID();
					float rating = preference.getValue();
					if (!this.AR.containsKey(userID)) {
						this.AR.put(userID, new Average(rating));
					} else {
						this.AR.get(userID).add(rating);
					}
					if (!this.AC.containsKey(itemID)) {
						this.AC.put(itemID, new Average(rating));
					} else {
						this.AC.get(itemID).add(rating);
					}
				}
			}

			iterate(nbMaxIterations);
			postProcess();
		}

		private void postProcess() throws TasteException {
			DataModel dataModel = getDataModel();
			LongPrimitiveIterator it = dataModel.getUserIDs();
			while (it.hasNext()) {
				long userID = it.nextLong();
				int g = this.Rho.get(userID).get();
				Average userRealAvg = new Average();
				Average userPredAvg = new Average();
				PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
				for (Preference preference : prefs) {
					long itemID = preference.getItemID();
					int h = this.Gamma.get(itemID).get();
					float rating = preference.getValue();
					userRealAvg.add(rating);
					float x = this.AR.get(userID).compute() + this.AC.get(itemID).compute() - this.ARC.get(g).compute()
							- this.ACC.get(h).compute() + this.ACOC.get(g).get(h).compute();
					userPredAvg.add(x);
				}
				this.bias.put(userID, userRealAvg.compute() - userPredAvg.compute());
			}
		}

		private int iterate(int iter) throws TasteException {

			DataModel dataModel = getDataModel();
			LongPrimitiveIterator itU;
			LongPrimitiveIterator itI;

			/* Repeat until convergence */
			int iterNb = 0;
			int nbChanged = 0;
			do {
//				log.info("Convergence loop: iteration #{}, previous rounds had {} changings", iterNb, nbChanged);
				nbChanged = 0;

				this.ACOC = new ArrayList<ArrayList<Average>>(this.k);
				this.ARC = new ArrayList<Average>(this.k);
				this.ACC = new ArrayList<Average>(this.l);
				for (int g = 0; g < this.k; g++) {
					this.ARC.add(new Average());
					ArrayList<Average> list = new ArrayList<Average>(this.l);
					for (int h = 0; h < this.l; h++) {
						list.add(new Average());
						if (g == 0) {
							this.ACC.add(new Average());
						}
					}
					this.ACOC.add(list);
				}

				/* Compute averages */
				itU = dataModel.getUserIDs();
				while (itU.hasNext()) {
					long userID = itU.nextLong();
					int g = this.Rho.get(userID).get();
					PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
					for (Preference preference : prefs) {
						long itemID = preference.getItemID();
						float rating = preference.getValue();
						int h = this.Gamma.get(itemID).get();
						Average avgRC = this.ARC.get(g);
						if (avgRC == null) {
							this.ARC.set(g, new Average(rating));
						} else {
							avgRC.add(rating);
						}
						Average avgCC = this.ACC.get(h);
						if (avgCC == null) {
							this.ACC.set(h, new Average(rating));
						} else {
							avgCC.add(rating);
						}
						Average avgCOC = this.ACOC.get(g).get(h);
						if (avgCOC == null) {
							this.ACOC.get(g).set(h, new Average(rating));
						} else {
							avgCOC.add(rating);
						}

					}
				}

				/* Update row assignment */
				itU = dataModel.getUserIDs();
				while (itU.hasNext()) {
					long userID = itU.nextLong();
					int curIdx = this.Rho.get(userID).get();
					int minIdx = curIdx;
					float min = Float.MAX_VALUE;
					for (int g = 0; g < this.k; g++) {
						float candidate = 0;
						PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
						for (Preference preference : prefs) {
							long itemID = preference.getItemID();
							float rating = preference.getValue();
							int h = this.Gamma.get(itemID).get();
							float x = rating - this.ACOC.get(g).get(h).compute() - this.AR.get(userID).compute()
									+ this.ARC.get(g).compute() - this.AC.get(itemID).compute()
									+ this.ACC.get(h).compute();
							candidate += x * x;
						}
						if (prefs.length() != 0 && candidate <= min) {
							min = candidate;
							minIdx = g;
						}
					}
					if (minIdx != curIdx) {
						nbChanged++;
						this.Rho.get(userID).set(minIdx);
					}
				}

				/* Update column assignment */
				itI = dataModel.getItemIDs();
				while (itI.hasNext()) {
					long itemID = itI.nextLong();
					int curIdx = this.Gamma.get(itemID).get();
					int minIdx = curIdx;
					float min = Float.MAX_VALUE;
					for (int h = 0; h < this.l; h++) {
						float candidate = 0;
						PreferenceArray prefs = dataModel.getPreferencesForItem(itemID);
						for (Preference preference : prefs) {
							long userID = preference.getUserID();
							float rating = preference.getValue();
							int g = this.Rho.get(userID).get();
							float x = rating - this.ACOC.get(g).get(h).compute() - this.AR.get(userID).compute()
									+ this.ARC.get(g).compute() - this.AC.get(itemID).compute()
									+ this.ACC.get(h).compute();
							candidate += x * x;
						}
						if (prefs.length() != 0 && candidate <= min) {
							min = candidate;
							minIdx = h;
						}
					}
					if (minIdx != curIdx) {
						nbChanged++;
						this.Gamma.get(itemID).set(minIdx);
					}
				}

				iterNb++;
			} while (iterNb < iter && nbChanged > 0);
			return iterNb;
		}

		private float getEstimate(long userID, long itemID) {
			double estimate;
			if (this.Rho.containsKey(userID)) {
				int g = this.Rho.get(userID).get();
				if (this.Gamma.containsKey(itemID)) {
					int h = this.Gamma.get(itemID).get();
					estimate = this.AR.get(userID).compute() + this.AC.get(itemID).compute() - this.ARC.get(g).compute()
							- this.ACC.get(h).compute() + this.ACOC.get(g).get(h).compute() + this.bias.get(userID);
				} else {
					estimate = this.AR.get(userID).compute();
				}
			} else {
				if (this.Gamma.containsKey(itemID)) {
					estimate = this.AC.get(itemID).compute();
				} else {
					estimate = 0;
				}
			}
			return (float) estimate;
		}

		private void splitUserCluster(int idx) throws TasteException {
			DataModel dataModel = getDataModel();
			int n = dataModel.getNumUsers();
			FastByIDMap<Index> newRho = new FastByIDMap<Index>(n);
			LongPrimitiveIterator it = dataModel.getUserIDs();
			while (it.hasNext()) {
				long userID = it.nextLong();
				int dest = this.Rho.get(userID).get();
				if (dest == idx && random.nextBoolean()) {
					dest = this.k;
				}
				newRho.put(userID, new Index(dest));
			}
			this.bakRho = this.Rho;
			this.bakGamma = this.Gamma;
			this.bakACC = this.ACC;
			this.bakACOC = this.ACOC;
			this.bakARC = this.ARC;
			this.Rho = newRho;
			this.k++;
		}

		private void splitItemCluster(int idx) throws TasteException {
			DataModel dataModel = getDataModel();
			int m = dataModel.getNumItems();
			FastByIDMap<Index> newGamma = new FastByIDMap<Index>(m);
			LongPrimitiveIterator it = dataModel.getItemIDs();
			while (it.hasNext()) {
				long itemID = it.nextLong();
				int dest = this.Gamma.get(itemID).get();
				if (dest == idx && random.nextBoolean()) {
					dest = this.l;
				}
				newGamma.put(itemID, new Index(dest));
			}
			this.bakRho = this.Rho;
			this.bakGamma = this.Gamma;
			this.bakACC = this.ACC;
			this.bakACOC = this.ACOC;
			this.bakARC = this.ARC;
			this.Gamma = newGamma;
			this.l++;
		}

		private void revertClusterSplit() {
			if (this.Rho != this.bakRho) {
				this.Rho = this.bakRho;
				this.k--;
			}
			if (this.Gamma != this.bakGamma) {
				this.Gamma = this.bakGamma;
				this.l--;
			}
			this.ACC = this.bakACC;
			this.ACOC = this.bakACOC;
			this.ARC = this.bakARC;
		}

		private TrainingError getTrainingError() throws TasteException {
			TrainingError error = new TrainingError(this.k, this.l);
			DataModel dataModel = getDataModel();
			LongPrimitiveIterator it = dataModel.getUserIDs();
			while (it.hasNext()) {
				long userID = it.nextLong();
				int g = this.Rho.get(userID).get();
				PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
				if (prefs != null) {
					for (Preference pref : prefs) {
						long itemID = pref.getItemID();
						int h = this.Gamma.get(itemID).get();
						float rating = pref.getValue();
						float x = rating - this.ACOC.get(g).get(h).compute() - this.AR.get(userID).compute()
								+ this.ARC.get(g).compute() - this.AC.get(itemID).compute() + this.ACC.get(h).compute();
						error.add(x, g, h);
					}
				}
			}
			return error;
		}

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
		return this.curRec.getEstimate(userID, itemID);
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

	private class Index {

		private int idx;

		Index(int n) {
			this.idx = n;
		}

		int get() {
			return this.idx;
		}

		void set(int n) {
			this.idx = n;
		}
	}

	private class MaxPair {

		private int i;
		private int j;
		private double value;

		MaxPair() {
			this.value = Double.MIN_VALUE;
			this.i = -1;
			this.j = -1;
		}

		void compare(int ii, int jj, double vv) {
			if (vv >= this.value) {
				this.i = ii;
				this.j = jj;
			}
		}

		int getRow() {
			return this.i;
		}

		int getCol() {
			return this.j;
		}

	}

	private class TrainingError {

		private final List<List<Average>> errors;
		private final Average total;
		private final int k;
		private final int l;

		TrainingError(int k, int l) {
			this.k = k;
			this.l = l;
			this.errors = new ArrayList<List<Average>>(this.k);
			for (int i = 0; i < this.k; i++) {
				List<Average> list = new ArrayList<Average>(this.l);
				for (int j = 0; j < this.l; j++) {
					list.add(new Average());
				}
				this.errors.add(list);
			}
			this.total = new Average();
		}

		void add(float err, int i, int j) {
			float x = err * err;
			this.errors.get(i).get(j).add(x);
			this.total.add(x);
		}

		double get(int i, int j) {
			return Math.sqrt(this.errors.get(i).get(j).compute());
		}

		double getTotal() {
			return Math.sqrt(this.total.compute());
		}

		MaxPair getMaximum() {
			MaxPair max = new MaxPair();
			for (int i = 0; i < this.k; i++) {
				for (int j = 0; j < this.l; j++) {
					double value = this.get(i, j);
					max.compare(i, j, value);
				}
			}
			return max;
		}

	}

}
