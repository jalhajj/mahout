package org.apache.mahout.cf.taste.impl.eval;

import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;

import com.google.common.base.Preconditions;

public class Fold {
	
	private final DataModel trainingSet;
	private final FastByIDMap<PreferenceArray> testingSet;
	
	public Fold(FastByIDMap<PreferenceArray> training, FastByIDMap<PreferenceArray> testing) {
		Preconditions.checkArgument(training != null, "training is null");
		Preconditions.checkArgument(testing != null, "testing is null");
		
		this.trainingSet = new GenericDataModel(training);
		this.testingSet = testing;
	}
	
	public DataModel getTraining() {
		return this.trainingSet;
	}
	
	public FastByIDMap<PreferenceArray> getTesting() {
		return this.testingSet;
	}

}
