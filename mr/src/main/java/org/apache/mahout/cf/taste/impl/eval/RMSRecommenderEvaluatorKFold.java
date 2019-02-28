package org.apache.mahout.cf.taste.impl.eval;

import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.Preference;

public final class RMSRecommenderEvaluatorKFold extends AbstractKFoldRecommenderEvaluator {

  private RunningAverage average;
  
  public RMSRecommenderEvaluatorKFold() {
	  super();
  }

  public RMSRecommenderEvaluatorKFold(long seed) {
	  super(seed);
  }

  @Override
  protected void reset() {
      average = new FullRunningAverage();
  }

  @Override
  protected void processOneEstimate(float estimatedPreference, Preference realPref) {
      double diff = realPref.getValue() - estimatedPreference;
      average.addDatum(diff * diff);
  }

  @Override
  protected double computeFinalEvaluation() {
      return Math.sqrt(average.getAverage());
  }

  @Override
  public String toString() {
      return "RMSRecommenderEvaluatorKFold";
  }
  
}

