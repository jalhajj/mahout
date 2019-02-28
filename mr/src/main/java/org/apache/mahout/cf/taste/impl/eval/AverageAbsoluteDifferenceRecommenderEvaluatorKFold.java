package org.apache.mahout.cf.taste.impl.eval;

import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.Preference;

public final class AverageAbsoluteDifferenceRecommenderEvaluatorKFold extends
    AbstractKFoldRecommenderEvaluator {
  
  private RunningAverage average;
  
  public AverageAbsoluteDifferenceRecommenderEvaluatorKFold() {
	  super();
  }
  
  public AverageAbsoluteDifferenceRecommenderEvaluatorKFold(long seed) {
	  super(seed);
  }
  
  @Override
  protected void reset() {
    average = new FullRunningAverage();
  }
  
  @Override
  protected void processOneEstimate(float estimatedPreference, Preference realPref) {
    average.addDatum(Math.abs(realPref.getValue() - estimatedPreference));
  }
  
  @Override
  protected double computeFinalEvaluation() {
    return average.getAverage();
  }
  
  @Override
  public String toString() {
    return "AverageAbsoluteDifferenceRecommenderEvaluatorKFold";
  }
  
}
