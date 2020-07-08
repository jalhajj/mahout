package org.apache.mahout.cf.taste.impl.eval;

import java.util.Iterator;
import java.util.Random;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.PredictionStatistics;
import org.apache.mahout.cf.taste.eval.FoldDataSplitter;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.recommender.COCLUSTRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

public final class KFoldRecommenderPredictionEvaluator {

	private static final Logger log = LoggerFactory.getLogger(KFoldRecommenderPredictionEvaluator.class);

	private final DataModel dataModel;
	private final FoldDataSplitter folds;

	public KFoldRecommenderPredictionEvaluator(DataModel dataModel, int nbFolds, Random random) throws TasteException {
		Preconditions.checkArgument(dataModel != null, "dataModel is null");
		Preconditions.checkArgument(nbFolds > 1, "nbFolds must be > 1");

		this.dataModel = dataModel;
		this.folds = new KFoldDataSplitter(this.dataModel, nbFolds, random);
	}

	public KFoldRecommenderPredictionEvaluator(DataModel dataModel, FoldDataSplitter splitter) throws TasteException {
		Preconditions.checkArgument(dataModel != null, "dataModel is null");
		Preconditions.checkArgument(splitter != null, "splitter is null");

		this.dataModel = dataModel;
		this.folds = splitter;
	}

	public PredictionStatistics evaluate(RecommenderBuilder recommenderBuilder) throws TasteException {

		Preconditions.checkArgument(recommenderBuilder != null, "recommenderBuilder is null");
		log.info("Beginning evaluation");

		RunningAverage mae = new FullRunningAverage();
		RunningAverage rmse = new FullRunningAverage();
		int noEst = 0;
		int total = 0;
		
		String info = null;

		Iterator<Fold> itF = this.folds.getFolds();
		int k = 0;
		while (itF.hasNext()) {

			Fold fold = itF.next();

			DataModel trainingModel = fold.getTraining();
			FastByIDMap<PreferenceArray> testPrefs = fold.getTesting();
			LongPrimitiveIterator it = fold.getUserIDs().iterator();

			Recommender recommender = recommenderBuilder.buildRecommender(trainingModel, fold);
			
			if (recommender instanceof COCLUSTRecommender) {
				COCLUSTRecommender rec = (COCLUSTRecommender) recommender;
				info = String.valueOf(rec.getTrainingError());
			}

			double smae = 0;
			double srmse = 0;
			int cnt = 0;

			while (it.hasNext()) {

				long userID = it.nextLong();
				PreferenceArray prefs = testPrefs.get(userID);
				if (prefs == null || prefs.length() == 0) {
					log.debug("Ignoring user {}", userID);
					continue; // Oops we excluded all prefs for the user -- just move on
				}

				for (Preference pref : prefs) {
					long itemID = pref.getItemID();
					float truth = pref.getValue();
					try {
						total++;
						Float pred = recommender.estimatePreference(userID, itemID);
						if (!pred.isNaN()) {
							double x = truth - pred;
							smae += Math.abs(x);
							srmse += x * x;
							cnt++;			
						} else {
							noEst++;
						}
					} catch (NoSuchUserException nsee) {
						noEst++;
						continue;
					} catch (NoSuchItemException nsie) {
						noEst++;
						continue;
					}
				}

			}

			double imae = Double.NaN;
			double irmse = Double.NaN;
			if (cnt > 0) {
				imae = smae / (double) (cnt);
				irmse = Math.sqrt(srmse / (double) (cnt));
			}
			mae.addDatum(imae);
			rmse.addDatum(irmse);
			
			log.info("Results for fold {} are: mae={}, rmse={}", k, imae, irmse);
			
			k++;

		}

		double noEstPer = noEst / (double) total; 
		
		log.info("Final results are: mae={}, rmse={}, noEstPer={}", mae.getAverage(), rmse.getAverage(), noEstPer);
		return new PredictionStatisticsImpl(mae.getAverage(), rmse.getAverage(), noEstPer, info);

	}

}
