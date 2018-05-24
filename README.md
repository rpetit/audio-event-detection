# Audio event detection
Work conducted as an intern at IRCAM. <br/> <br/>
<div align="left">
  <img src="http://www2.atiam.ircam.fr/wp-content/uploads/2011/07/logoircam.jpg" width="30%"><br><br>
</div>

## `pip` installation 
Run the following command :

```
pip install -e git+https://github.com/romainpe/audio-event-detection#egg=evdetect
```

## Example

Detection of a bird's song with a hidden Markov model. <br/>

From the examples directory, run bird.py to get the following output :

![bird](bird.png)

This will also output a .wav file in examples/results where only the segments matching the event model 
have been preserved. </br>

See [this page](https://romainpe.github.io/audio-event-detection) for more examples.

## References
[1] A. Bietti, online learning for audio clustering and segmentation Master thesis, 2014 <br/><br/>
[2] A. Cont, a coupled duration-focused architecture for realtime music to score alignment, IEEE Transactions on Pattern 
Analysis and Machine Intelligence, 2010 <br/><br/>
[3] Y. Matsubara, Y. Sakurai, N. Ueda and M. Yoshikawa, fast and exact monitoring of co-evolving data streams, IEEE 
International Conference on Data Mining, 2014 <br/><br/>
[4] L. Rabiner, a tutorial on hidden Markov models and selected applications in speech recognition, Proceedings of the 
IEEE, 1989