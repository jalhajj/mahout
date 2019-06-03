package org.apache.mahout.cf.taste.impl.recommender;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;

import com.google.common.base.Preconditions;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Average;
import org.apache.mahout.cf.taste.impl.common.Bicluster;
import org.apache.mahout.cf.taste.impl.common.Biclustering;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.RatingSGDFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class COCLUSTRecommender extends AbstractRecommender {
	
	private static double NO_REGULARIZATION = -1;

	private final Random random;
	private final int k;
	private final int l;
	private final int nbMaxIterations;
	private final double lambda;
	private ArrayList<ArrayList<Average>> ACOC;
	private ArrayList<Average> ARC;
	private ArrayList<Average> ACC;
	private FastByIDMap<Average> AR;
	private FastByIDMap<Average> AC;
	private FastByIDMap<Index> Rho;
	private FastByIDMap<Index> Gamma;
	private FastByIDMap<Float> bias;
	private final RefreshHelper refreshHelper;

	private ArrayList<ArrayList<Recommender>> recs;

	private static final Logger log = LoggerFactory.getLogger(COCLUSTRecommender.class);

	public COCLUSTRecommender(DataModel dataModel, int nbUserClusters, int nbItemClusters, int maxIter,
			CandidateItemsStrategy strategy, double lambda) throws TasteException {
		super(dataModel, strategy);
		this.random = RandomUtils.getRandom();
		this.k = nbUserClusters;
		this.l = nbItemClusters;
		this.nbMaxIterations = maxIter;
		this.lambda = lambda;
		refreshHelper = new RefreshHelper(new Callable<Object>() {
			@Override
			public Object call() throws TasteException {
				train();
				return null;
			}
		});
		refreshHelper.addDependency(getDataModel());
		init();
	}
	
	public COCLUSTRecommender(DataModel dataModel, int nbUserClusters, int nbItemClusters, int maxIter,
			CandidateItemsStrategy strategy) throws TasteException {
		super(dataModel, strategy);
		this.random = RandomUtils.getRandom();
		this.k = nbUserClusters;
		this.l = nbItemClusters;
		this.nbMaxIterations = maxIter;
		this.lambda = NO_REGULARIZATION;
		refreshHelper = new RefreshHelper(new Callable<Object>() {
			@Override
			public Object call() throws TasteException {
				train();
				return null;
			}
		});
		refreshHelper.addDependency(getDataModel());
		init();
	}

	public COCLUSTRecommender(DataModel dataModel, int nbUserClusters, int nbItemClusters, int maxIter)
			throws TasteException {
		super(dataModel);
		this.random = RandomUtils.getRandom();
		this.k = nbUserClusters;
		this.l = nbItemClusters;
		this.nbMaxIterations = maxIter;
		this.lambda = NO_REGULARIZATION;
		refreshHelper = new RefreshHelper(new Callable<Object>() {
			@Override
			public Object call() throws TasteException {
				train();
				return null;
			}
		});
		refreshHelper.addDependency(getDataModel());
		init();
	}

	public void init() throws TasteException {

		log.info("COCLUST Recommender with k={} and l={}", this.k, this.l);

		DataModel dataModel = getDataModel();
		int n = dataModel.getNumUsers();
		int m = dataModel.getNumItems();

		this.AR = new FastByIDMap<Average>(n);
		this.AC = new FastByIDMap<Average>(m);
		this.Rho = new FastByIDMap<Index>(n);
		this.Gamma = new FastByIDMap<Index>(m);
		this.bias = new FastByIDMap<Float>(n);

		log.debug("Done with initialization, about to start training");
		train();
	}

	void randomInit() throws TasteException {
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

		DataModel dataModel = getDataModel();
		LongPrimitiveIterator itU;

		/* Randomly initialize biclusters */
		log.debug("Starting with random biclusters");
		randomInit();

		/* Pre-compute AR and AC */
		log.debug("Pre-computing rows and columns averages");
		itU = dataModel.getUserIDs();
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

		iterate(this.nbMaxIterations);

		/* Post processing to compute bias */
		itU = dataModel.getUserIDs();
		while (itU.hasNext()) {
			long userID = itU.nextLong();
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

		/* WIP */

//		List<List<FastByIDMap<List<Preference>>>> datasets = new ArrayList<List<FastByIDMap<List<Preference>>>>(this.k);
//		for (int g = 0; g < this.k; g++) {
//			List<FastByIDMap<List<Preference>>> list = new ArrayList<FastByIDMap<List<Preference>>>(this.l);
//			for (int h = 0; h < this.l; h++) {
//				FastByIDMap<List<Preference>> ratings = new FastByIDMap<List<Preference>>();
//				list.add(ratings);
//			}
//			datasets.add(list);
//		}
//		
//		itU = dataModel.getUserIDs();
//		while (itU.hasNext()) {
//			long userID = itU.nextLong();
//			int g = this.Rho.get(userID).get();
//			PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
//			for (Preference preference : prefs) {
//				long itemID = preference.getItemID();
//				int h = this.Gamma.get(itemID).get();
//				FastByIDMap<List<Preference>> ratings = datasets.get(g).get(h);
//				if (!ratings.containsKey(userID)) {
//					ratings.put(userID, new ArrayList<Preference>());
//				}
//				ratings.get(userID).add(preference);
//			}
//		}
//		
//		this.recs = new ArrayList<ArrayList<Recommender>>(this.k);
//		for (int g = 0; g < this.k; g++) {
//			ArrayList<Recommender> list = new ArrayList<Recommender>(this.l);
//			for (int h = 0; h < this.l; h++) {
//				
//				FastByIDMap<List<Preference>> ratings = datasets.get(g).get(h);
//				FastByIDMap<PreferenceArray> userData = new FastByIDMap<PreferenceArray>(ratings.size());
//				itU = ratings.keySetIterator();
//				while (itU.hasNext()) {
//					long userID = itU.nextLong();
//					PreferenceArray a = new GenericUserPreferenceArray(ratings.get(userID).size());
//					int id = 0;
//					for (Preference pref : ratings.get(userID)) {
//						a.set(id, new GenericPreference(userID, pref.getItemID(), pref.getValue()));
//						id++;
//					}
//					userData.put(userID, a);
//				}
//				
//				Factorizer fact = new RatingSGDFactorizer(dataModel, 18, this.nbMaxIterations);
//				Recommender rec = new SVDRecommender(new GenericDataModel(userData), fact, this.candidateItemsStrategy);
//				list.add(rec);
//			}
//			this.recs.add(list);
//		}

	}

	public Biclustering<Long> getBiclustering() throws TasteException {

		Biclustering<Long> bicl = new Biclustering<Long>();

		// Init structure
		List<List<Bicluster<Long>>> biclusters = new ArrayList<List<Bicluster<Long>>>(this.k);
		for (int g = 0; g < this.k; g++) {
			List<Bicluster<Long>> list = new ArrayList<Bicluster<Long>>(this.l);
			for (int h = 0; h < this.l; h++) {
				Bicluster<Long> b = new Bicluster<Long>();
				list.add(b);
				bicl.add(b);
			}
			biclusters.add(list);
		}

		// Fill with ids
		DataModel dataModel = this.getDataModel();
		LongPrimitiveIterator it = dataModel.getUserIDs();
		while (it.hasNext()) {
			long userID = it.nextLong();
			int g = this.Rho.get(userID).get();
			for (int h = 0; h < this.l; h++) {
				biclusters.get(g).get(h).addUser(userID);
			}
		}
		it = dataModel.getItemIDs();
		while (it.hasNext()) {
			long itemID = it.nextLong();
			int h = this.Gamma.get(itemID).get();
			for (int g = 0; g < this.k; g++) {
				biclusters.get(g).get(h).addItem(itemID);
			}
		}

		return bicl;
	}

	public int iterate(int iter) throws TasteException {

		DataModel dataModel = getDataModel();
		LongPrimitiveIterator itU;
		LongPrimitiveIterator itI;

		/* Repeat until convergence */
		int iterNb = 0;
		int nbChanged = 0;
		do {
			log.debug("Convergence loop: iteration #{}, previous rounds had {} changings", iterNb, nbChanged);
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
			log.debug("Compute biclusters averages");
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
			log.debug("Update row assignments");
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
								+ this.ARC.get(g).compute() - this.AC.get(itemID).compute() + this.ACC.get(h).compute();
						candidate += x * x;
					}
					
					if (this.lambda != NO_REGULARIZATION) {
						double max = Double.MIN_VALUE;
						for (int h = 0; h < this.l; h++) {
							double x = this.ACOC.get(g).get(h).getCount();
							if (x >= max) {
								max = x;
							}
						}
						if (max > Double.MIN_VALUE) {
							candidate += this.lambda * max;
						}
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
			log.debug("Update column assignments");
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
								+ this.ARC.get(g).compute() - this.AC.get(itemID).compute() + this.ACC.get(h).compute();
						candidate += x * x;
					}
					
					if (this.lambda != NO_REGULARIZATION) {
						double max = Double.MIN_VALUE;
						for (int g = 0; g < this.k; g++) {
							double x = this.ACOC.get(g).get(h).getCount();
							if (x >= max) {
								max = x;
							}
						}
						if (max > Double.MIN_VALUE) {
							candidate += this.lambda * max;
						}
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
		double estimate;
		if (this.Rho.containsKey(userID)) {
			int g = this.Rho.get(userID).get();
			if (this.Gamma.containsKey(itemID)) {
				int h = this.Gamma.get(itemID).get();
				estimate = this.AR.get(userID).compute() + this.AC.get(itemID).compute() - this.ARC.get(g).compute()
						- this.ACC.get(h).compute() + this.ACOC.get(g).get(h).compute();// + this.bias.get(userID);
			} else {
				estimate = this.AR.get(userID).compute();
			}
		} else {
			if (this.Gamma.containsKey(itemID)) {
				estimate = this.AC.get(itemID).compute();
			} else {
				estimate = Float.NaN;
			}
		}
		return (float) estimate;
	}

	public double getTrainingError() throws TasteException {

		double sum = 0;
		int cnt = 0;
		
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
					sum += x * x;
					cnt++;
				}
			}
		}
		return Math.sqrt(sum / (double) cnt);

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
		refreshHelper.refresh(alreadyRefreshed);
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

}
