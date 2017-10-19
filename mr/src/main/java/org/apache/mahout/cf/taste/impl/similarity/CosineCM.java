package org.apache.mahout.cf.taste.impl.similarity;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.impl.common.AbstractCountMinSketch;
import org.apache.mahout.cf.taste.impl.common.DoubleCountMinSketch;

import com.google.common.base.Preconditions;

import java.util.HashMap;


public final class CosineCM extends AbstractSimilarity {
  
  private final double epsilon;
  private final double delta;
  private final HashMap<Long, DoubleCountMinSketch> sketches;

  /**
   * @throws IllegalArgumentException if {@link DataModel} does not have preference values
   */
  public CosineCM(DataModel dataModel, double eps, double del) throws TasteException {
    this(dataModel, Weighting.UNWEIGHTED, eps, del);
  }

  /**
   * @throws IllegalArgumentException if {@link DataModel} does not have preference values
   */
  public CosineCM(DataModel dataModel, Weighting weighting, double eps, double del) throws TasteException {
    super(dataModel, weighting, false);
    epsilon = eps;
    delta = del;
    sketches = new HashMap<Long, DoubleCountMinSketch>(dataModel.getNumUsers());
    Preconditions.checkArgument(dataModel.hasPreferenceValues(), "DataModel doesn't have preference values");
  }
  
  public DoubleCountMinSketch exportCMProfile(long userID) throws TasteException {
    
    DoubleCountMinSketch cm = sketches.get(userID);
    
    if (cm == null) {
      
      try{
        cm = new DoubleCountMinSketch(delta, epsilon);
      } catch(AbstractCountMinSketch.CMException ex) {
        throw new TasteException("CountMinSketch error:" + ex.getMessage());
      }
      
      DataModel dataModel = getDataModel();
      PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
      int length = prefs.length();
      for (int i = 0; i < length; i++) {
        long index = prefs.getItemID(i);
        double x = prefs.getValue(i);
        cm.update(index, x);
      }
      sketches.put(userID, cm);
    }
    
    return cm;
  }

  @Override
  double computeResult(int n, double sumXY, double sumX2, double sumY2, double sumXYdiff2) {
    if (n == 0) {
      return Double.NaN;
    }
    double denominator = Math.sqrt(sumX2) * Math.sqrt(sumY2);
    if (denominator == 0.0) {
      // One or both parties has -all- the same ratings;
      // can't really say much similarity under this measure
      return Double.NaN;
    }
    return sumXY / denominator;
  }
  
  @Override
  public double userSimilarity(long userID1, long userID2) throws TasteException {
    
    DoubleCountMinSketch cm1 = exportCMProfile(userID1);
    DoubleCountMinSketch cm2 = exportCMProfile(userID2);
    
    double result = DoubleCountMinSketch.cosine(cm1, cm2);
    
    if (!Double.isNaN(result)) {
      result = normalizeWeightResult(result, 1, 0);
    }
    return result;
    
  }
  
  @Override
  public final double itemSimilarity(long itemID1, long itemID2) throws TasteException {
    
    DataModel dataModel = getDataModel();
    PreferenceArray xPrefs = dataModel.getPreferencesForItem(itemID1);
    PreferenceArray yPrefs = dataModel.getPreferencesForItem(itemID2);
    int xLength = xPrefs.length();
    int yLength = yPrefs.length();
    
    if (xLength == 0 || yLength == 0) {
      return Double.NaN;
    }
    
    long xIndex, yIndex;
    double x, y;
    
    DoubleCountMinSketch cm1, cm2;
    
    try{
      
      cm1 = new DoubleCountMinSketch(delta, epsilon);
      cm2 = new DoubleCountMinSketch(delta, epsilon);
      
    } catch(AbstractCountMinSketch.CMException ex) {
      throw new TasteException("CountMinSketch error:" + ex.getMessage());
    }
    
    for (int i = 0; i < xLength || i < yLength; i++) {
      if (i < xLength) {
        xIndex = xPrefs.getUserID(i);
        x = xPrefs.getValue(i);
        cm1.update(xIndex, x);
      }
      if (i < yLength) {
        yIndex = yPrefs.getUserID(i);
        y = yPrefs.getValue(i);
        cm2.update(yIndex, y);
      }
    }
    
    double result = DoubleCountMinSketch.cosine(cm1, cm2);
    int count = xLength > yLength ? yLength : xLength; // Set minimum, but not important I think
    
    if (!Double.isNaN(result)) {
      result = normalizeWeightResult(result, count, 0);
    }
    return result;
  
  }

}
