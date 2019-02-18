package org.apache.mahout.cf.taste.impl.common;

import java.util.ArrayList;
import java.util.Iterator;

public class Biclustering<E> {
	
	private ArrayList<Bicluster<E>> biclusters; 
	
	Biclustering() {
		this.biclusters = new ArrayList<Bicluster<E>>();
	}
	
	void add(Bicluster<E> b) {
		this.biclusters.add(b);
	}
	
	public int size() {
		return this.biclusters.size();
	}
	
	public Iterator<Bicluster<E>> iterator() {
		return this.biclusters.iterator();
	}
	
	public String toString() {
		return this.biclusters.toString();
	}

}
