package org.apache.mahout.cf.taste.impl.common;

import java.util.List;

public class Bicluster {
	
	private List<Long> users;
	private List<Long> items;
	
	Bicluster(List<Long> u, List<Long> j) {
		this.users = u;
		this.items = j;
	}
	
	void addUser(long user) {
		this.users.add(user);
	}
	
	void removeUser(long user) {
		this.users.remove(user);
	}
	
	void addItem(long item) {
		this.items.add(item);
	}
	
	void removeItem(long item) {
		this.items.remove(item);
	}
	
	int getNbUsers() {
		return this.users.size();
	}
	
	int getNbItems() {
		return this.items.size();
	}

}
