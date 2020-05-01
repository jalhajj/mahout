package org.apache.mahout.cf.taste.eval;

import java.util.List;

import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;

public interface PerUserStatistics {
  
  double getRMSE(long userID);
  double getMAE(long userID);
  double getPrecision(long userID);
  double getRecall(long userID);
  double getNormalizedDiscountedCumulativeGain(long userID);
  double getOther(long userID);
  String getMisc(long userID);
  double getItemCoverage();
  List<Double> getHitsFrom(long userID);
  void addRMSE(long userID, double rmse);
  void addMAE(long userID, double mae);
  void addPrecision(long userID, double precision);
  void addRecall(long userID, double recall);
  void addNDCG(long userID, double ndcg);
  void addOther(long userID, double other);
  void addHitsFrom(long userID, List<Double> hits);
  void addMisc(long userID, String str);
  void addItemCoverage(double icov);
  LongPrimitiveIterator getUserIDs();
  
}
