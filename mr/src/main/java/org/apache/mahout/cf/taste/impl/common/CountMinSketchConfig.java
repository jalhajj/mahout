package org.apache.mahout.cf.taste.impl.common;

import java.lang.Math;
import java.lang.ClassNotFoundException;
import java.io.Serializable;
import java.io.IOException;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.AbstractCountMinSketch;
import org.apache.mahout.cf.taste.impl.common.DoubleCountMinSketch;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.commons.lang3.StringUtils;

public class CountMinSketchConfig implements Serializable {
  
  private static final transient Logger log = LoggerFactory.getLogger(CountMinSketchConfig.class);
  
  private transient int MAX_WIDTH = 10000;
  private transient int MAX_DEPTH = 25;
  
  private final double gamma; // Deniability wished
  private final double betaC; // Error bound wished for cosine query
  private final double betaP; // Error bound wished for point query
  private EDResult result;
  
  
  /** Class to serialize the result of the computation */
  class EDResult implements Serializable {

    private final double delta;
    private final double epsilon;
    
    EDResult(double d, double e) {
      delta = d;
      epsilon = e;
    }
    
  }
  
  
  /**
   *  @param    g   gamma-deniability condition value
   *  @param    e   Error bound value
   */
  public CountMinSketchConfig(double g, double bc, double bp) {
    gamma = g;
    betaC = bc;
    betaP = bp;
    result = null;
  }
  
  
  /** Configure the count-min sketch delta and epsilon parameters
   *  to ensure a given level of deniability and to ensure a
   *  probabilistic error bound on the accuracy
   * 
   *  Must be called before getDelta() and getEpsilon()
   * 
   *  @param  dataModel     Dataset
   *  @param  datasetPath   Path of the dataset file, used as dataset
   *                        identifier for serialization
   * 
   *  @throws TasteException 
   */
  public void configure(DataModel dataModel, String datasetPath) throws TasteException {
    String datasetName = StringUtils.substringBefore(datasetPath.replace("/", "-"), ".");
    String path = "ser/" + datasetName + "_gamma_" + gamma + "_betaC_" + betaC + "_betaP_" + betaP + ".ser";
    log.info("Try to find {} file, check if already computed", path);
    try {
      /* Check if computation was already serialized in a previous run */
      FileInputStream fileIn = new FileInputStream(path);
      ObjectInputStream in = new ObjectInputStream(fileIn);
      result = (EDResult) in.readObject(); // If found, retrieve the result
      log.info("Found file, already computed, retrieved results delta={} and epsilon={}",
                getDelta(), getEpsilon());
      in.close();
      fileIn.close();
    } catch(IOException ex) {
      /* If not found, compute and save the result for next time */
      log.info("Found nothing, let's compute then");
      computeConfig(dataModel);
      save(path);
    } catch(ClassNotFoundException ex) {
      log.error("ClassNotFoundException: {}", ex.getMessage());
    }
    
  }
  
  
  /** Serialize the result of the configuration
   * 
   *  @param  path  name of the file where to save the result
   */
  private void save(String path) {
    try {
      FileOutputStream fileOut = new FileOutputStream(path);
      ObjectOutputStream out = new ObjectOutputStream(fileOut);
      out.writeObject(result);
      log.info("Result saved for future experiments");
      out.close();
      fileOut.close();
    } catch(IOException ex) {
      log.error("IOException: {}", ex.getMessage());
    }
  }
    
  
  /** Compute epsilon and delta to meet the conditions required
   *  
   *  @throws TasteException    If not possible to meet the conditions
   */ 
  private void computeConfig(DataModel dataModel) throws TasteException {
    
    LongPrimitiveIterator it = dataModel.getUserIDs();
    int n = 0;
    int I = dataModel.getNumItems();
    
    /* NOTE: For now, consider all users, but in the future, we may want to not
     * bother about those with huge profiles. For instance, the width chosen
     * could double because of ONE user..
     */
    /* Find the maximum number of items in a user profile */
    while (it.hasNext()) {
      long userID = it.next();
      int profileSize = dataModel.getPreferencesFromUser(userID).length();
      n = Math.max(n, profileSize);
    }
    
    /* Try to find a solution with branch and bound / binary search techniques */
    int minWidth = getWidthForCosineError(n);
    Integer minDepth = 1;
    int w;
    for (w = minWidth; w < MAX_WIDTH; w++) {
      minDepth = getDepthForPointError(w, n);
      if (minDepth == null) {
        continue;
      } else {
        minDepth = getDepthForDeniability(I, n, w, minDepth);
        if (minDepth == null) {
          continue;
        } else {
          break;
        }
      }
    }
    
    /* Check if a solution was found */
    if (w == MAX_WIDTH) {
      throw new TasteException("Not possible to meet conditions, try with different parameters");
    }
    
    /* Compute the chosen parameters */
    double epsilon = Math.exp(1) / (double) minWidth;
    double delta = Math.exp(- (double) minDepth);
    result = new EDResult(delta, epsilon);
    log.info("Parameters chosen: width={} (epsilon={}), depth={} (delta={})",
              minWidth, epsilon, minDepth, delta);
  }
  
  
  /** Compute gamma-deniability with the approximation formula
   * 
   * @param   u   Total number of keys
   * @param   n   Number of keys inserted in the sketch
   * @param   w   Width of the sketch
   * @param   d   Depth of the sketch
   * 
   * @return  gamma-deniability value
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
  
  
  /** Get the maximum depth to ensure a given deniability for a given user
   * 
   *  @param  I         Total number of items
   *  @param  n         Maximum number of keys to insert in a sketch
   *  @param  width     Width of the count-min sketch
   *  @param  minDepth  Minimum value required for the depth
   * 
   *  @return   Minimum depth >= minDepth that ensures the deniability condition is met
   *            or null if not possible
   * 
   *  @throws TasteException    If not possible to meet the condition
   */
  private Integer getDepthForDeniability(int I, int n, int width, int minDepth) throws TasteException {
    
    double currentDen;
    int currentDepth;
    int da = minDepth, db = MAX_DEPTH;
    
    /* Check if min depth is small enough to find a solution, not possible otherwise */
    currentDen = gammaDeniability(I, n, width, da);
    if (currentDen < gamma) {
      return null;
    }
    
    /* Start binary search */
    while (da < (db - 1)) {
      log.debug("For deniability, search between {} and {}", da, db);
      currentDepth = (da + db) / 2;
      currentDen = gammaDeniability(I, n, width, da);
      if (currentDen < gamma) {
        db = currentDepth;
      } else {
        da = currentDepth;
      }
    }
    log.debug("For gamma={} and width={}, depth selected to meet deniability condition is {}", gamma, width, da);
    return new Integer(da);
  }
  
  
  /** Compute the expected value for the percentage of collisions in a row
   * 
   *  @param  width   width of the count-min sketch
   *  @param  nb      number of elements inserted
   * 
   *  @return   expected percentage of collisions in a row
   */
  private double expectedPercentageColisions(int width, int nb) {
    double w = (double) width;
    double n = (double) nb;
    return (n - w * (1 - Math.pow(1 - 1 / w, n))) / n;
  }
  
  
  /** Get the width to ensure a given error bound is met for a given user
   * 
   *  @param    n   Maximum number of keys to insert in a sketch
   * 
   *  @return   Minimum width to ensure the error condition for cosine query is met
   * 
   *  @throws TasteException    If not possible to meet the condition
   */
  private int getWidthForCosineError(int n) throws TasteException {
    
    double currentError;
    int currentWidth;
    int wa = 1, wb = MAX_WIDTH;
    
    /* Check if max width is big enough to find a solution, not possible otherwise */
    currentError = expectedPercentageColisions(wb, n);
    if (currentError > betaC) {
      String msg = String.format("Not possible to meet cosine error <= %g condition, would require a width greater than %d",
                                betaC, wb);
      throw new TasteException(msg);
    }
    
    /* Start binary search */
    while (wa < (wb - 1)) {
      log.debug("For cosine error, search between {} and {}", wa, wb);
      currentWidth = (wa + wb) / 2;
      currentError = expectedPercentageColisions(currentWidth, n);
      if (currentError > betaC) {
        wa = currentWidth;
      } else {
        wb = currentWidth;
      }
    }
    log.debug("For cosine error={}, width selected to meet condition is {}", betaC, wb);
    return wb;
  }
  
  
  /** Compute the probability that a point query does not return the exact value
   * 
   *  @param  width   Width of the count-min sketch
   *  @param  depth   Depth of the count-min sketch
   *  @param  nb      Number of elements inserted
   * 
   *  @return   expected percentage of collisions in a row
   */
  private double notExactPointQueryProba(int width, int depth, int nb) {
    if (nb == 0) { return 0.0; }
    double w = (double) width;
    double d = (double) depth;
    double n = (double) nb;
    return Math.pow(1 - Math.pow(1 - 1 / w, nb - 1), d);
  }
  
  
  /** Get the depth to ensure a given error bound is met for a given user
   * 
   *  @param    n   Maximum number of keys to insert in a sketch
   * 
   *  @return   Minimum depth to ensure the error condition for point query is met
   *            or null if not possible
   * 
   *  @throws TasteException    If not possible to meet the condition
   */
  private Integer getDepthForPointError(int width, int n) throws TasteException {
    
    double currentError;
    int currentDepth;
    int da = 1, db = MAX_DEPTH;
    
    /* Check if max depth is big enough to find a solution, not possible otherwise */
    currentError = notExactPointQueryProba(width, db, n);
    if (currentError > betaP) {
      return null;
    }
    
    /* Start binary search */
    while (da < (db - 1)) {
      log.debug("For point error, search between {} and {}", da, db);
      currentDepth = (da + db) / 2;
      currentError = notExactPointQueryProba(width, currentDepth, n);
      if (currentError > betaP) {
        da = currentDepth;
      } else {
        db = currentDepth;
      }
    }
    log.debug("For point error={}, width selected to meet condition is {}", betaP, db);
    return new Integer(db);
  }
  
  
  /** Return delta parameter
   * 
   * @return  delta parameter
   * 
   * @throws  TasteException    If configure method was not called first
   */
  public double getDelta() throws TasteException {
    if (result == null) {
      throw new TasteException("delta is null, call configure method first");
    } else
    return result.delta;
  }
  
  
  /** Return epsilon parameter
   * 
   * @return  epsilon parameter
   * 
   * @throws  TasteException    If configure method was not called first
   */
  public double getEpsilon() throws TasteException {
    if (result == null) {
      throw new TasteException("epsilon is null, call configure method first");
    } else
    return result.epsilon;
  }
  
  
  
}
