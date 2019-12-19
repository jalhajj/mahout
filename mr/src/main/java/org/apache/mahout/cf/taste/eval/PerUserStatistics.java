package org.apache.mahout.cf.taste.eval;

import java.util.List;

import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;

public interface PerUserStatistics {
  
  double getRMSE(long userID);
  double getMAE(long userID);
  double getPrecision(long userID);
  double getRecall(long userID);
  double getNormalizedDiscountedCumulativeGain(long userID);
  List<Double> getHitsFrom(long userID);
  void addRMSE(long userID, double rmse);
  void addMAE(long userID, double mae);
  void addPrecision(long userID, double precision);
  void addRecall(long userID, double recall);
  void addNDCG(long userID, double ndcg);
  void addHitsFrom(long userID, List<Double> hits);
  LongPrimitiveIterator getUserIDs();
  
}
