package org.apache.mahout.cf.taste.impl.common;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Bimax {

	private static Logger log = LoggerFactory.getLogger(Bimax.class);

	private DataModel dataModel;
	private Biclustering<Long> bicl;
	private ArrayList<Long> userMap;
	private ArrayList<Long> itemMap;
	private FastByIDMap<Integer> invUserMap;
	private FastByIDMap<Integer> invItemMap;
	private int n;
	private int m;
	private int minN;
	private int minM;
	private float threshold;

	public Bimax(DataModel data, float thres) throws TasteException {
		this(data, thres, 1, 1);
	}

	/**
	 * Performs a biclustering based on Bimax exhaustive divide and conquer search
	 * 
	 * @param data        User item matrix
	 * @param thres       Threshold to obtain binary ratings
	 * @param minUserSize minimum number of users in a bicluster
	 * @param minItemSize minimum number of items in a bicluster
	 * @throws TasteException
	 */
	public Bimax(DataModel data, float thres, int minUserSize, int minItemSize) throws TasteException {
		LongPrimitiveIterator it;
		int i;
		this.dataModel = data;
		this.bicl = new Biclustering<Long>();
		this.threshold = thres;
		this.n = data.getNumUsers();
		this.m = data.getNumItems();
		this.minN = minUserSize;
		this.minM = minItemSize;
		this.userMap = new ArrayList<Long>(this.n);
		this.itemMap = new ArrayList<Long>(this.m);
		this.invUserMap = new FastByIDMap<Integer>(this.n);
		this.invItemMap = new FastByIDMap<Integer>(this.m);

		IntBicluster bicluster = new IntBicluster(this.m);

		it = data.getUserIDs();
		i = 0;
		while (it.hasNext()) {
			long userID = it.nextLong();
			this.userMap.add(userID);
			this.invUserMap.put(userID, i);
			bicluster.addUser(i);
			i++;
		}

		it = data.getItemIDs();
		i = 0;
		while (it.hasNext()) {
			long itemID = it.nextLong();
			this.itemMap.add(itemID);
			this.invItemMap.put(itemID, i);
			bicluster.addItem(i);
			i++;
		}

		log.info("Starting divide-and-conquer search");
		conquer(bicluster, null);
		log.info("Done with the search, found {} biclusters", this.bicl.size());

	}

	private void conquer(Bicluster<Integer> bicluster, List<Integer> mandatory) throws TasteException {

		/* Check if empty */
		if (bicluster.isEmpty()) {
			return;
		}

		/* Check size */
		if (bicluster.getNbUsers() < this.minN || bicluster.getNbItems() < this.minM) {
			return;
		}

		/*
		 * Check if end of recursion (full of ones and has at least one of mandatory
		 * columns)
		 */
		boolean onlyOnes = true;
		boolean hasManda = mandatory == null ? true : false;
		int curRow = 0;
		Iterator<Integer> itU = bicluster.getUsers();
		while (onlyOnes && itU.hasNext()) {
			curRow = itU.next();
			long userID = this.userMap.get(curRow);

			/* Filter out some cases to avoid useless checking */
			if (this.dataModel.getPreferencesFromUser(userID).length() < bicluster.getNbItems()) {
				onlyOnes = false;
			}

			Iterator<Integer> itI = bicluster.getItems();
			while (onlyOnes && itI.hasNext()) {
				int j = itI.next();
				Float rating = this.dataModel.getPreferenceValue(userID, this.itemMap.get(j));
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
			}

		} else {

			/* Divide and conquer, use curRow as template */
			IntBicluster bcU = new IntBicluster(this.m);
			IntBicluster bcV = new IntBicluster(this.m);
			ArrayList<Integer> CV = new ArrayList<Integer>();

			long curUserID = this.userMap.get(curRow);

			Iterator<Integer> itI = bicluster.getItems();
			while (itI.hasNext()) {
				int j = itI.next();
				Float rating = this.dataModel.getPreferenceValue(curUserID, this.itemMap.get(j));
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
				long userID = this.userMap.get(i);
				boolean inU = false;
				boolean inV = false;
				for (Preference pref : this.dataModel.getPreferencesFromUser(userID)) {
					if (inU && inV) {
						break;
					}
					long itemID = pref.getItemID();
					int j = this.invItemMap.get(itemID);
					if (bicluster.containsItem(j)) {
						float rating = pref.getValue();
						Float ratingRef = this.dataModel.getPreferenceValue(curUserID, itemID);
						if (rating >= this.threshold && (ratingRef != null && ratingRef >= this.threshold)) {
							inU = true;
						}
						if (rating >= this.threshold && (ratingRef == null || ratingRef < this.threshold)) {
							inV = true;
						}
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
