package org.apache.mahout.cf.taste.impl.common;

import java.util.List;
import java.util.BitSet;

public class IntBicluster extends Bicluster<Integer> {
	
	private BitSet isItemIn;
	
	IntBicluster(List<Integer> u, List<Integer> j) {
		super(u, j);
		this.isItemIn = null;
	}
	
	IntBicluster() {
		super();
		this.isItemIn = null;
	}
	
	IntBicluster(int size) {
		super();
		this.isItemIn = new BitSet(size);
	}
	
	@Override
	void addItem(Integer item) {
		super.addItem(item);
		if (this.isItemIn != null) {
			this.isItemIn.set(item);
		}
	}
	
	@Override
	void removeItem(Integer item) {
		super.removeItem(item);
		if (this.isItemIn != null) {
			this.isItemIn.set(item, false);
		}
	}
	
	@Override
	public boolean containsItem(Integer item) {
		if (isItemIn != null) {
			return this.isItemIn.get(item);
		} else {
			return super.containsItem(item);
		}
	}

}
