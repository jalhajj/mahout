package org.apache.mahout.cf.taste.eval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.eval.Fold;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

public class ChronologicalDataSplitter implements FoldDataSplitter {
	
	private static final Logger log = LoggerFactory.getLogger(ChronologicalDataSplitter.class);
	
	private final List<Fold> folds;
	private final double trainingPercentage;
	
	public ChronologicalDataSplitter(DataModel dataModel, double trainingPercentage) throws TasteException {
		Preconditions.checkArgument(dataModel != null, "dataModel is null");
		Preconditions.checkArgument(trainingPercentage > 0 && trainingPercentage < 1, "trainingPercentage must be > 0 and < 1");
		
		log.info("Initialiazing chronological fold");
		
		this.folds = new ArrayList<Fold>(1);
		this.trainingPercentage = trainingPercentage;
		
		int n = dataModel.getNumUsers();
		
		List<Long> allTimestamps = new ArrayList<Long>();
		
		LongPrimitiveIterator it = dataModel.getUserIDs();
		while (it.hasNext()) {
			long userID = it.nextLong();
			for (Preference pref: dataModel.getPreferencesFromUser(userID)) {
				long ts = dataModel.getPreferenceTime(userID, pref.getItemID());
				allTimestamps.add(ts);
			}
		}
		
		long tsThreshold = getPercentile(allTimestamps, this.trainingPercentage);
		
		FastByIDMap<FastByIDMap<Long>> trainingTimestamps = new FastByIDMap<FastByIDMap<Long>>(n);

		FastByIDMap<PreferenceArray> training = new FastByIDMap<PreferenceArray>();
		FastByIDMap<PreferenceArray> testing = new FastByIDMap<PreferenceArray>();

		// Split the dataModel into K folds per user
		it = dataModel.getUserIDs();
		while (it.hasNext()) {
			long userID = it.nextLong();
			FastByIDMap<Long> timestamps = new FastByIDMap<Long>();
			splitOneUsersPrefs(training, timestamps, testing, userID, dataModel, tsThreshold);
			trainingTimestamps.put(userID, timestamps);
		}
		this.folds.add(new Fold(training, trainingTimestamps, testing));
		log.info("{} folds", this.folds.size());
	}

	@Override
	public Iterator<Fold> getFolds() {
		return this.folds.iterator();
	}
	
	private void splitOneUsersPrefs(FastByIDMap<PreferenceArray> training, FastByIDMap<Long> trainingTimestamps, FastByIDMap<PreferenceArray> testing,
			long userID, DataModel dataModel, long tsThreshold) throws TasteException {

		PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);

		List<Preference> train = new ArrayList<Preference>();
		List<Preference> test = new ArrayList<Preference>();
		
		Iterator<Preference> it = prefs.iterator();
		while (it.hasNext()) {
			Preference pref = it.next();
			long itemID = pref.getItemID();
			Preference newPref = new GenericPreference(userID, itemID, pref.getValue());
			long ts = dataModel.getPreferenceTime(userID, itemID);
			if (ts <= tsThreshold) {
				train.add(newPref);
				trainingTimestamps.put(itemID, ts);
			} else {
				test.add(newPref);
			}
		}

		if (!train.isEmpty()) {
			training.put(userID, new GenericUserPreferenceArray(train));
		}
		if (!test.isEmpty()) {
			testing.put(userID, new GenericUserPreferenceArray(test));
		}

	}
	
	private static long getPercentile(List<Long> values, double percentile) {
        Collections.sort(values);
        int index = (int) Math.ceil(percentile * (double) values.size());
        index = Math.max(index, 0);
        return values.get(index - 1);
    }
	
	class ChronologicalPrefComparator implements Comparator<Preference> {
		
		private final DataModel datamodel;
		private final long userID;
		
		ChronologicalPrefComparator(DataModel datamodel, long userID) {
			this.datamodel = datamodel;
			this.userID = userID;
		}

		@Override
		public int compare(Preference pref1, Preference pref2) {
			long itemID1 = pref1.getItemID();
			long itemID2 = pref2.getItemID();
			long ts1, ts2;
			try {
				ts1 = this.datamodel.getPreferenceTime(this.userID, itemID1);
				ts2 = this.datamodel.getPreferenceTime(this.userID, itemID2);
				return Long.compare(ts1, ts2);
			} catch (TasteException e) {
				return 0;
			}
		}
		
	}

}
