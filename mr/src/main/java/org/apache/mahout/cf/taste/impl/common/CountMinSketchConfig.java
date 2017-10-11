package org.apache.mahout.cf.taste.impl.common;

import java.lang.Math;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.AbstractCountMinSketch;
import org.apache.mahout.cf.taste.impl.common.DoubleCountMinSketch;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CountMinSketchConfig {
  
  private static final Logger log = LoggerFactory.getLogger(CountMinSketchConfig.class);
  
  private int MAX_WIDTH = 1000;
  
  
  private double delta = 0;
  private double epsilon = 0;
  
  private final double gamma; // Deniability wished
  private final double error; // Error bound wished
  private int DEPTH;          // Depth
  
  public CountMinSketchConfig(double g, double e, int d) {
    gamma = g;
    error = e;
    DEPTH = d;
  }
  
  /** Configure the count-min sketch delta and epsilon parameters
   *  to ensure a given level of deniability and to ensure that
   *  a bound is met on the point-query error
   * 
   *  Must be called before getDelta() and getEpsilon()
   */
  public void configure(DataModel dataModel) throws TasteException {
    
    int width = 0;
    LongPrimitiveIterator it = dataModel.getUserIDs();
    while (it.hasNext()) {
      long userID = it.next();
      int w = getWidthForError(dataModel, userID, getWidthForDeniability(dataModel, userID));
      width = Math.max(width, w);
      log.debug("Width {} chosen for user {}, current max width is {}", w, userID, width);
    }
    
    log.debug("Width chosen is {}, now check if error is still ok", width);
    
    // Check if the width chosen is ok for error on all users
    it = dataModel.getUserIDs();
    while (it.hasNext()) {
      long userID = it.next();
      double e = computeError(dataModel, userID, width);
      if (e > error) {
        log.warn("Error for user {} is required to be < to {} but is {} with width chosen {}",
                  userID, error, e, width);
      }
    }
    
    epsilon = Math.exp(1) / (double) width;
    delta = Math.exp(- (double) DEPTH);
    log.info("Parameters chosen: width {} (epsilon {}), depth {} (delta {})",
              width, epsilon, DEPTH, delta);
  }
  
  /** Compute gamma-deniability thanks to the approximation formula
   * 
   * @param   u   total number of keys
   * @param   n   number of keys inserted in the sketch
   * @param   w   width of the sketch
   * @param   d   depth of the sketch
   * 
   * @return  gamma-deniability
   */
  private double gammaDeniability(int u, int n, int w, int d) {
    double U = (double) u;
    double N = (double) n;
    double W = (double) w;
    double D = (double) d;
    
    double cardDiff = U - N;
    double invWidth = 1 / W;
    double p = 1 - Math.pow(1 - invWidth, N);
    double q = 1 - Math.pow(1 - invWidth / p, cardDiff * p);
    return Math.pow(q, D);
  }
  
  /** Compute the error for a given width
   *  
   * @return  max error over all items where error for one item
   *          is computed as :
   *          | real value - point query value | / real value
   */
  private double computeError(DataModel dataModel, long userID, int width) throws TasteException {
    
    try {
      
      // Create the CM and insert items
      DoubleCountMinSketch cm = new DoubleCountMinSketch(width, DEPTH);
      PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
      int length = prefs.length();
      for (int i = 0; i < length; i++) {
        long index = prefs.getItemID(i);
        double x = prefs.getValue(i);
        cm.update(index, x);
      }
      
      // Compute error
      double maxError = 0;
      for (int i = 0; i < length; i++) {
        long index = prefs.getItemID(i);
        double x = prefs.getValue(i);
        double y = cm.get(index);
        double e = Math.abs(x - y) / x;
        maxError = Math.max(maxError, e);
        log.debug("For width {}, user {}, item {}, real rating is {} and point query returns {}: error is {}, max error for this user is {}",
                  width, userID, index, x, y, e, maxError);
      }
      return maxError;
      
    } catch(AbstractCountMinSketch.CMException ex) {
      throw new TasteException("CountMinSketch error:" + ex.getMessage());
    }
    
  }
  
  /** Get the width required to ensure a given deniability */
  private int getWidthForDeniability(DataModel dataModel, long userID) throws TasteException {
    
    int currentWidth = 2 * MAX_WIDTH;
    double currentGamma;
    do {
      currentWidth = currentWidth / 2;
      if (currentWidth == 1) {
        throw new TasteException("Not possible to meet deniability condition");
      }
      currentGamma = gammaDeniability(dataModel.getNumItems(), dataModel.getPreferencesFromUser(userID).length(),
                                      currentWidth, DEPTH);
      log.debug("For width {} and user {} ({} items in profile among {} items), deniability is {} while required one is {}",
                currentWidth, userID, dataModel.getPreferencesFromUser(userID).length(), dataModel.getNumItems(),
                currentGamma, gamma);
    } while (currentGamma < gamma);
    
    return currentWidth;
      
  }
  
  /** Get the width to ensure a given error bound is met */
  private int getWidthForError(DataModel dataModel, long userID, int maxWidth) throws TasteException {
    
    int currentWidth = 0;
    double currentMaxError;
    do {
      currentWidth++;
      if (currentWidth > maxWidth) {
        throw new TasteException("Not possible to meet error condition");
      }
      currentMaxError = computeError(dataModel, userID, currentWidth);
      log.debug("For width {} and user {}, max error is {}, required is {}",
                currentWidth, userID, currentMaxError, error);
    } while (currentMaxError > error);
    
    return currentWidth;
      
  }
  
  /** Return delta parameter
   * @return  delta parameter
   */
  public double getDelta() throws TasteException {
    if (delta == 0) {
      throw new TasteException("delta is null, call configure method first");
    } else
    return delta;
  }
  
  /** Return epsilon parameter
   * @return  epsilon parameter
   */
  public double getEpsilon() throws TasteException {
    if (epsilon == 0) {
      throw new TasteException("epsilon is null, call configure method first");
    } else
    return epsilon;
  }
  
  
  
}
