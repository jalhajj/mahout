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

public class CountMinSketchConfig implements Serializable {
  
  private static final transient Logger log = LoggerFactory.getLogger(CountMinSketchConfig.class);
  
  private transient int MAX_DEPTH = 25;
  private transient int MIN_DEPTH = 1;
  
  private final double q; // Privacy/Accuracy trade-off parameter
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
  public CountMinSketchConfig(double q_) {
    q = q_;
    result = null;
  }
  
  
  /** Configure the count-min sketch delta and epsilon parameters
   * 
   *  Must be called before getDelta() and getEpsilon()
   * 
   *  @param  dataModel     Dataset
   *  @param  datasetName   Name of the dataset file, used as dataset
   *                        identifier for serialization
   * 
   *  @throws TasteException 
   */
  public void configure(DataModel dataModel, String datasetName) throws TasteException {

    String path = "ser/" + datasetName + "_q_" + q + ".ser";
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
    
  
  /** Compute epsilon and delta by solving optimization problem
   *  
   *  @throws TasteException    If not possible to meet the conditions
   */ 
  private void computeConfig(DataModel dataModel) throws TasteException {
    
    LongPrimitiveIterator it = dataModel.getUserIDs();
    int n = 0;
    int u = dataModel.getNumItems();
    
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
    
    int bestWidth = 0;
    int bestDepth = 0;
    double bestMax = 0;
    for (int d = MIN_DEPTH; d < MAX_DEPTH; d++) {
			for (int w = d; w <= n; w++) {
				double x = Fmeasure(w, d, n, u, q);
				if (x >= bestMax) {
					bestWidth = w;
					bestDepth = d;
					bestMax = x;
				}
			}
		}
    
    /* Check if a solution was found */
    if (bestWidth == 0 && bestDepth == 0) {
      throw new TasteException("No solution found (this should not happen)");
    }
    
    /* Compute the chosen parameters */
    double epsilon = Math.exp(1) / (double) bestWidth;
    double delta = Math.exp(- (double) bestDepth);
    result = new EDResult(delta, epsilon);
    log.info("Parameters chosen: width={} (epsilon={}), depth={} (delta={})",
              bestWidth, epsilon, bestDepth, delta);
  }
  
  
  /** Compute the inserted probability
   * 
   * @param   w   Width of the sketch
   * @param   d   Depth of the sketch
   * @param   n   Number of keys inserted in the sketch
   * @param   u   Total number of keys
   * 
   * @return  inserted probability value
   */
  public static double probaInserted(int w, int d, int n, int u) {
		double W = (double) w;
    double D = (double) d;
    double N = (double) n;
    double U = (double) u;
    
    double falseP = Math.pow(1 - Math.pow(1 - 1 / W, N), D);
    return N / (N + falseP * (U - N));
  }
  
  
  /** Compute the not-exact retrieve probability
   * 
   * @param   w   Width of the sketch
   * @param   d   Depth of the sketch
   * @param   n   Number of keys inserted in the sketch
   *
   * @return	the probability for a point-query 
   * 					to NOT retrieve the exact value
   */
  public static double probaNotExactRetrieve(int w, int d, int n) {
		double W = (double) w;
    double D = (double) d;
    double N = (double) n;
    
		return Math.pow(1 - Math.pow(1 - 1 / W, N), D);
	}
	
	
	/** Function the optimization problem seeks to maximize
	 *  Inspired from precision / recall f-measure
	 * 
   * @param   w   Width of the sketch
   * @param   d   Depth of the sketch
   * @param   n   Number of keys inserted in the sketch
   * @param   u   Total number of keys
   * @param		q		Accuracy / privacy trade-off parameter
   * 
   * @return	f-measure value
   */
	public static double Fmeasure(int w, int d, int n, int u, double q) {
    double beta = 1 - probaNotExactRetrieve(w, d, n);
    double p = 1 - probaInserted(w, d, n, u);
    if (beta == 0 || p == 0) {
			return 0;
		} else {
			double q2 = Math.pow(q, 2);
			return (1 + 2) * beta * p / (q2 * beta + p);
		}
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
