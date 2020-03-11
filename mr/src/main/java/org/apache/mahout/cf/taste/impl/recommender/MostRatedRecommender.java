/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf.taste.impl.recommender;

import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Counter;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * <p>
 * Like {@link ItemAverageRecommender}, except that estimated preferences are adjusted for the users' average
 * preference value. For example, say user X has not rated item Y. Item Y's average preference value is 3.5.
 * User X's average preference value is 4.2, and the average over all preference values is 4.0. User X prefers
 * items 0.2 higher on average, so, the estimated preference for user X, item Y is 3.5 + 0.2 = 3.7.
 * </p>
 */
public final class MostRatedRecommender extends AbstractRecommender {
  
  private static final Logger log = LoggerFactory.getLogger(MostRatedRecommender.class);
  
  private final FastByIDMap<Counter> itemCounts;
  private final ReadWriteLock buildCountsLock;
  private final RefreshHelper refreshHelper;
  
  public MostRatedRecommender(DataModel dataModel) throws TasteException {
    super(dataModel);
    this.itemCounts = new FastByIDMap<>();
    this.buildCountsLock = new ReentrantReadWriteLock();
    this.refreshHelper = new RefreshHelper(new Callable<Object>() {
      @Override
      public Object call() throws TasteException {
    	  buildCounts();
        return null;
      }
    });
    refreshHelper.addDependency(dataModel);
    buildCounts();
  }
  
  public MostRatedRecommender(DataModel dataModel, CandidateItemsStrategy strategy) throws TasteException {
	    super(dataModel, strategy);
	    this.itemCounts = new FastByIDMap<>();
	    this.buildCountsLock = new ReentrantReadWriteLock();
	    this.refreshHelper = new RefreshHelper(new Callable<Object>() {
	      @Override
	      public Object call() throws TasteException {
	    	  buildCounts();
	        return null;
	      }
	    });
	    refreshHelper.addDependency(dataModel);
	    buildCounts();
	  }
  
  @Override
  public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems)
    throws TasteException {
    Preconditions.checkArgument(howMany >= 1, "howMany must be at least 1");
    log.debug("Recommending items for user ID '{}'", userID);

    PreferenceArray preferencesFromUser = getDataModel().getPreferencesFromUser(userID);
    FastIDSet possibleItemIDs = getAllOtherItems(userID, preferencesFromUser, includeKnownItems);

    TopItems.Estimator<Long> estimator = new Estimator();

    List<RecommendedItem> topItems = TopItems.getTopItems(howMany, possibleItemIDs.iterator(), rescorer,
      estimator);

    log.debug("Recommendations are: {}", topItems);
    return topItems;
  }
  
  @Override
  public float estimatePreference(long userID, long itemID) throws TasteException {
    DataModel dataModel = getDataModel();
    Float actualPref = dataModel.getPreferenceValue(userID, itemID);
    if (actualPref != null) {
      return actualPref;
    }
    return doEstimatePreference(userID, itemID);
  }
  
  private float doEstimatePreference(long userID, long itemID) {
	  return Float.NaN;
  }

  private void buildCounts() throws TasteException {
    try {
      buildCountsLock.writeLock().lock();
      DataModel dataModel = getDataModel();
      LongPrimitiveIterator it = dataModel.getItemIDs();
      while (it.hasNext()) {
        long itemID = it.nextLong();
        PreferenceArray prefs = dataModel.getPreferencesForItem(itemID);
        int size = prefs.length();
        this.itemCounts.put(itemID, new Counter(size));
      }
    } finally {
      buildCountsLock.writeLock().unlock();
    }
  }
  
  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }
  
  @Override
  public String toString() {
    return "MostRatedRecommender";
  }
  
  private final class Estimator implements TopItems.Estimator<Long> {
    
    private Estimator() {
    }
    
    @Override
    public double estimate(Long itemID) {
      return itemCounts.get(itemID).get();
    }
  }
  
}
