f = FluidModel();
f.slicer = FluidModelSlicer.novelty(1, 7, 0.25);
//f.build("/home/bgola/workspace/samples/BAILEFUNK2020/toflucoma/");
f.build("/home/bgola/tmp/sopr/");
d = f.export;
d=d.reject{|el| el.isEmpty}
d.size


d[100].mfcc.plot

d[10].keys
f.durations[10]

d.do {|el,idx|
	d[idx].duration = f.durations[idx];
}

d[0].pitch

d[0].spectralShape
(
v = d
//.select {|el| el.spectralShape[0] < -30 }
//.select {|el| el.pitch[1] > 0.5 }
.select {|el| el.duration[0] > 0.05 and: {el.duration[0] < 0.07 } }
//.select {|el| el.loudness[0] < -50 } // and: {el.loudness[0] < -10} }
//.sort {|a,b| a.pitch[0] < b.pitch[0] }
.sort {|a,b| a.loudness[0] < b.loudness[0] }
//.sort {|a,b| a.spectralShape[0] < b.spectralShape[0]}
;
v.size.postln;
)

v.sum{ |el| el.duration[0] } / 60

d[0].loudness

r.size
(
SynthDef(\playsl, {arg out=0, sliceStart, sliceDur, atk=0, rel=0;
	Out.ar(0, PlayBuf.ar(2, f.buffer, 1, 1, sliceStart) * Env.new([0,1,1,0], [atk,1-(atk+rel),rel]).ar(2, 1, sliceDur))
}).add

)

(
b = ();
d.do {|e|
	b[e.duration[0].round(0.1)] = b[e.duration[0].round(0.1)].add(e);
	//b[e.loudness[0].asInteger] = b[e.loudness[0].asInteger].add(e);
}
)

c = [2.5,0.1,0.1,0.1,0.5,0.1,0.2] // 0.1,0.2,0.4,0.5,0.8];
c = [-60, -30, -20, -60, -50];
b[0.1]

(
fork {
	loop {
		n = c.collect {|dur| b[dur].choose };
		n.do{|slice|
			//var slice = b[dur].choose;
			var dur = slice.duration[0];
			Synth(\playsl, [
				\sliceStart, slice.duration[1],
				\sliceDur, slice.duration[0],
				\atk, 0.2,
				\rls, 0.2
			]);
			//slice.duration[0].wait;
			dur.postln.wait;
		}
	}
}
)
r.size

(
fork {
	v.do {|slice|
		Synth(\playsl, [
			\sliceStart, slice.duration[1],
			\sliceDur, slice.duration[0],
			\atk, 0.1,
			\rls, 0.1
		]);
		slice.pitch[0].postln;
		(slice.duration[0]*0.5).wait;
	}
}
)

(
fork {
	r.do {|slice|
		var total = 0;
		while ({total < 0.4}) {
			Synth(\playsl, [\sliceStart, slice.duration[1], \sliceDur, slice.duration[0]]);
			slice.loudness[0].postln;
			(slice.duration[0]).wait;
			total = total + slice.duration[0];
		}
	}
}
)



d[0].keys