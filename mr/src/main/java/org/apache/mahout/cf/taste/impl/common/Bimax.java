package org.apache.mahout.cf.taste.impl.common;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.collections.CollectionUtils;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Bimax {
	
	private static final Logger log = LoggerFactory.getLogger(Bimax.class);
	
	private DataModel dataModel;
	private Biclustering<Long> bicl;
	private ArrayList<Long> userMap;
	private ArrayList<Long> itemMap;
	private int n;
	private int m;
	private float threshold;
	
	/** Performs a biclustering based on Bimax exhaustive divide and conquer search
	 * 
	 * @param data User item matrix
	 * @param thres Threshold to obtain binary ratings
	 * @throws TasteException
	 */
	public Bimax(DataModel data, float thres) throws TasteException {
		LongPrimitiveIterator it;
		int i;
		this.dataModel = data;
		this.bicl = new Biclustering<Long>();
		this.threshold = thres;
		this.n = data.getNumUsers();
		this.m = data.getNumItems();
		this.userMap = new ArrayList<Long>(this.n);
		this.itemMap = new ArrayList<Long>(this.m);
		
		Bicluster<Integer> bicluster = new Bicluster<Integer>();
		
		it = data.getUserIDs();
		i = 0;
		while (it.hasNext()) {
			long userID = it.nextLong();
			this.userMap.add(userID);
			bicluster.addUser(i);
			i++;
		}
		
		it = data.getItemIDs();
		i = 0;
		while (it.hasNext()) {
			long itemID = it.nextLong();
			this.itemMap.add(itemID);
			bicluster.addItem(i);
			i++;
		}
		
		conquer(bicluster, null);
		
	}
	
	private void conquer(Bicluster<Integer> bicluster, List<Integer> mandatory) throws TasteException {
		
		log.info("Current bicluster is {}, mandatory columns were {}", bicluster.toString(), mandatory == null ? "none" : mandatory.toString());
		
		/* Check if empty */
		if (bicluster.isEmpty()) {
			return;
		}
		
		/* Check if end of recursion (full of ones and has at least one of mandatory columns) */
		boolean onlyOnes = true;
		boolean hasManda = mandatory == null ? true : false;
		int curRow = 0;
		Iterator<Integer> itU = bicluster.getUsers();
		while (onlyOnes && itU.hasNext()) {
			curRow = itU.next();
			Iterator<Integer> itI = bicluster.getItems();
			while (onlyOnes && itI.hasNext()) {
				int j = itI.next();
				Float rating = this.dataModel.getPreferenceValue(this.userMap.get(curRow), this.itemMap.get(j));
				if (rating != null && rating >= this.threshold) {
					if (mandatory != null && mandatory.contains(j)) {
						hasManda = true;
					}
				} else {
					onlyOnes = false;
				}
			}
		}
		
		if (onlyOnes) {
			
			if (hasManda) {
				/* Add bicluster */
				Bicluster<Long> bc = new Bicluster<Long>();
				itU = bicluster.getUsers();
				while (itU.hasNext()) {
					int i = itU.next();
					bc.addUser(this.userMap.get(i));
				}
				Iterator<Integer> itI = bicluster.getItems();
				while (itI.hasNext()) {
					int j = itI.next();
					bc.addItem(this.itemMap.get(j));
				}
				this.bicl.add(bc);
				log.info("Added bicluster {}, mandatory columns were {}", bc.toString(), mandatory == null ? "none" : mandatory.toString());
			}
			
		} else {
			
			/* Divide and conquer, use curRow as template */
			Bicluster<Integer> bcU = new Bicluster<Integer>();
			Bicluster<Integer> bcV = new Bicluster<Integer>();
			ArrayList<Integer> CV = new ArrayList<Integer>();
			Iterator<Integer> itI = bicluster.getItems();
			while (itI.hasNext()) {
				int j = itI.next();
				Float rating = this.dataModel.getPreferenceValue(this.userMap.get(curRow), this.itemMap.get(j));
				if (rating != null && rating >= this.threshold) {
					bcU.addItem(j);
				} else {
					CV.add(j);
				}
				bcV.addItem(j);
			}
			itU = bicluster.getUsers();
			while (itU.hasNext()) {
				int i = itU.next();
				boolean inU = false;
				boolean inV = false;
				itI = bicluster.getItems();
				while ((!inU  || !inV) && itI.hasNext()) {
					int j = itI.next();
					Float rating = this.dataModel.getPreferenceValue(this.userMap.get(i), this.itemMap.get(j));
					Float ratingRef = this.dataModel.getPreferenceValue(this.userMap.get(curRow), this.itemMap.get(j));
					if (rating != null && ratingRef != null && rating >= this.threshold && ratingRef >= this.threshold) {
						inU = true;
					}
					if (rating != null && rating >= this.threshold && (ratingRef == null || ratingRef < this.threshold)) {
						inV = true;
					}
				}
				if (inU) {
					bcU.addUser(i);
				}
				if (inV) {
					bcV.addUser(i);
				}
			}
			
			conquer(bcU, mandatory);
			conquer(bcV, CV);
			
		}
	}
	
	public Biclustering<Long> get() {
		return this.bicl;
	}

}
