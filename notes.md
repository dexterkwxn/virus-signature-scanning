gems article thing
------------------
idea:
1. find 2 byte values from each signature that are distinct from each other
2. for each 2 byte, map the 4 bytes before to an array s.t. A[2-byte] = 4-byte 
 -- identify the signature somehow (another array or 2d or something)
3. use GPU as high-speed filter by scanning input
  a. for each 2 bytes in input that matches any signature's 2 bytes,
  b. compare with input's prev 4 bytes 
  c. if match, send to CPU to do complete check (filter vibe check success)


step 3a, parallelize by chunking the input of each file into blocks (experiment w size)


step 3c, wait for all threads to be done -> group all possible matches, then send back to CPU. 
--(to save on time spent copying data)


brute force
----------
idea:
1. scan for possible matches on GPU, save index in array
 -- check with first byte of every signature for each input index
2. get array of all possible matches for every signature -- A[signature][starting_idx]
3. run for each signature, partition A[signature] and check using GPU if they match.
