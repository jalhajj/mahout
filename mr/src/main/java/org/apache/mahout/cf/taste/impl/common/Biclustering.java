package org.apache.mahout.cf.taste.impl.common;

import java.util.ArrayList;

public class Biclustering<E> {
	
	private ArrayList<Bicluster<E>> biclusters; 
	
	Biclustering() {
		this.biclusters = new ArrayList<Bicluster<E>>();
	}
	
	void add(Bicluster<E> b) {
		this.biclusters.add(b);
	}
	
	Bicluster<E> get(int i) {
		return this.biclusters.get(i);
	}
	
	int size() {
		return this.biclusters.size();
	}
	
	public String toString() {
		return this.biclusters.toString();
	}

}
