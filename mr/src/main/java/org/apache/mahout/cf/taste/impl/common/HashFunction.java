package org.apache.mahout.cf.taste.impl.common;

import java.math.BigInteger;
import java.util.Random;

import gnu.trove.list.array.TLongArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class HashFunction {
  
  private static final Logger log = LoggerFactory.getLogger(HashFunction.class);
  
  private BigInteger bigPrime = new BigInteger("9223372036854775783");
    
  /* Store some random parameters that should be the same for everyone
   * Parameters are generated in a lazy way when asked for
   */
  private static int index = -1;
  private static TLongArrayList randomParamA = new TLongArrayList();
  private static TLongArrayList randomParamB = new TLongArrayList();
  
  /* Initliaze a PRNG with the time seed
   * Another seed can be given using `initSeet(long s)` method
   */
  private static long seed = System.currentTimeMillis();
  private static Random rand = new Random(seed);
  
  /* Store parameters for hash computations */
  private final BigInteger a;
  private final BigInteger b;
  private final BigInteger w;
  
  
  /** Initialize the PRNG with a given seed
   * 
   *  @param  s   seed
   */
  public static void initSeed(long s) {
    seed = s;
    rand = new Random(seed);
  }
  
  
  /** Return a hash function for a given iteration index and width
   * 
   *  @param  iteration   Used to choose the parameters of the hash function
   *  @param  width       Range: a key will be hashed into {0, .., width - 1}
   */
  HashFunction(int iteration, int size) {
    
    /* Check if enough random parameters already generated, otherwise generate some more */
    if (iteration >= index) {
      for (int i = index + 1; i <= iteration; i++) {
        long a = rand.nextLong();
        long b = rand.nextLong();
        randomParamA.set(i, a);
        randomParamB.set(i, b);
      }
      index = iteration;
    }
    
    /* Store the parameters chosen */
    a = BigInteger.valueOf(randomParamA.get(iteration));
    b = BigInteger.valueOf(randomParamB.get(iteration));
    w = BigInteger.valueOf(size);
    
  }
  
  
  /** Hashes a key and returns an integer */
  int hash(long key) {
    BigInteger k = BigInteger.valueOf(key);
    return a.multiply(k).add(b).mod(bigPrime).mod(w).intValue();
  }
  
  
}
