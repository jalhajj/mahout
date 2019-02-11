package org.apache.mahout.cf.taste.impl.common;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Bicluster<E> {
	
	private List<E> users;
	private List<E> items;
	
	Bicluster(List<E> u, List<E> j) {
		this.users = u;
		this.items = j;
	}
	
	Bicluster() {
		this.users = new ArrayList<E>();
		this.items = new ArrayList<E>();
	}
	
	void addUser(E user) {
		this.users.add(user);
	}
	
	void removeUser(E user) {
		this.users.remove(user);
	}
	
	void addItem(E item) {
		this.items.add(item);
	}
	
	void removeItem(E item) {
		this.items.remove(item);
	}
	
	int getNbUsers() {
		return this.users.size();
	}
	
	int getNbItems() {
		return this.items.size();
	}
	
	Iterator<E> getUsers() {
		return this.users.iterator();
	}
	
	Iterator<E> getItems() {
		return this.items.iterator();
	}
	
	boolean isEmpty() {
		return this.users.isEmpty()  || this.items.isEmpty();
	}
	
	public String toString() {
		return this.users.toString() + "x" + this.items.toString();
	}

}
