package org.apache.mahout.cf.taste.impl.common;

import java.math.BigInteger;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class HashFunction {
  
  private static final Logger log = LoggerFactory.getLogger(HashFunction.class);
  
  private BigInteger bigPrime = new BigInteger("9223372036854775783");
  
  private static String[] randomParamA = {
    "3855483969251045550", "2274948550203382259", "8086346051022946365", "3742599363140121417",
    "8127506216481886778", "8992508990183509581", "6824159666599056129", "1515839073414842035",
    "3528950968555460028", "3742666563140121417", "8086685051022946685", "8926282854497893686",
    "7549048658687782964", "2254963801332704518", "4662970043116588076", "1164415423486499490",
    "8596154617854201450", "35867659355547039",   "7491907009982470321", "353950969553360028",
    "3742666563140908304"};
                                
  private static String[] randomParamB = {
    "8708688407112891618", "7360455785926943020", "8825528156902469898", "7810309941732699355",
    "2097246509310169464", "238488925168692289" , "260263731934835712",  "688270280519291225",
    "2179209161542291414", "7816665941732699355", "6855528156906859898", "4390665439843151325",
    "8145312472828294149", "2847806146481835937", "3995608470880119485", "947734926901625025",
    "4850723540680713533", "35867659355547039",   "7491907009982470321", "2079208161542291414",
    "9083045941732699355"};
   
  private final BigInteger a;
  private final BigInteger b;
  private final BigInteger w;
  
  HashFunction(int iteration, int size) {
    
    a = new BigInteger(randomParamA[iteration]);
    b = new BigInteger(randomParamB[iteration]);
    w = BigInteger.valueOf(size);
    
  }
  
  int hash(long key) {
    BigInteger k = BigInteger.valueOf(key);
    return a.multiply(k).add(b).mod(bigPrime).mod(w).intValue();
  }
  
  
}
