package org.apache.mahout.cf.taste.impl.common;

public class Counter {
	
	private int value;
	
	public Counter() {
		this.value = 0;
	}
	
	public Counter(int n) {
		this.value = n;
	}
	
	public int get() {
		return this.value;
	}
	
	public void reset() {
		this.value = 0;
	}
	
	public void incr() {
		incr(1);
	}
	
	public void incr(int n) {
		this.value += n;
	}
	
	public void decr() {
		decr(1);
	}
	
	public void decr(int n) {
		this.value = Math.max(0, this.value - n);
	}

}
