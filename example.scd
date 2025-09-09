
s.options.numOutputBusChannels = 8
f = FluidModel();
f.slicer = FluidModelSlicer.novelty(1,15, 0.25);
f.build("/home/bgola/workspace/CloudGarden/EchoMaterial/soundfiles");

e = f.export

f.store("/home/bgola/workspace/CloudGarden/EchoMaterial/slices_novelty_15")
f.load("/home/bgola/workspace/CloudGarden/EchoMaterial/slices_novelty_15")
File.use("/home/bgola/workspace/CloudGarden/EchoMaterial/slices_novelty_15/slices_novelty_15_exported.scd", "w", {|f| f.write(e.asCompileString)})
e = ("/home/bgola/workspace/CloudGarden/EchoMaterial/slices_novelty_15/slices_novelty_15_exported.scd").load

e

e

f.load("/home/bgola/workspace/CloudGarden/EchoMaterial/slices_novelty_18")

f.labelset
f.build("/home/bgola/workspace/samples/addtv/");
f.scaler
f.store("/tmp/test")
f.load("/tmp/test")
d = f.export;
e
File.use("/tmp/test/exported.scd", "w", {|f| f.write(d.asCompileString)})
d = ("/tmp/test/exported.scd").load

FluidWaveform(f.buffer, f.slices)

d = FluidDataSet(s);
FluidNormalize(s).fitTransform(f.datasets.mfcc, d, {"done".postln})
FluidPCA(s, 2).fitTransform(f.datasets.merged, d, {"done".postln})
FluidPCA(s, 2).fitTransform(d, d, {"done".postln})
FluidPCA(s, 3).fitTransform(d, d, {"done".postln})
FluidNormalize(s).fitTransform(d, d, {"done".postln})
t = FluidKDTree(s, 1);
t.fit(d);

~point = Buffer.alloc(s, 2);
~res = Buffer.alloc(s, 35);
~last = nil;
~amp = 0.2;
f.buffer.play
f.loader.files
(
d.dump({|dic| defer {
	p = FluidPlotter(dict: dic, mouseMoveAction: {|plotter, x,y, mod, btNum, clickCnt|
		if (clickCnt.isNil) {
			~point.setn(0, x, 1, y);
			t.kNearest(~point, 1, {|value|
				value.asString.postln;
				//if (value!=~last){
				~last = value ;
				f.datasets.merged.getPoint(value, ~res, { ~res.getn(30,5, {|vals| vals.postln;
					//Ndef(\player).set(\pos, vals[1], \end, vals[2]);
					//Ndef(\player).set(\posn, vals[1]/f.buffer.numFrames, \dura, (vals[2]-vals[1])/f.buffer.sampleRate);
					//Ndef(\player).set(\t_trig, 1);
					Synth(\playsl, [
						\sliceStart, vals[1], //duration[1],
						\sliceDur, (vals[2] - vals[1])/f.buffer.sampleRate, //slice.duration[0],
						\atk, 1.0.rand,
						\rls, 1.0.rand,
						\pan, 1.0.rand2, //(idx/v.size * 0.5).sin.postln,
						\amp, ~amp,
						\out, 0,
					]);
			})});
				//};
			});
		};
	});
}});
//fork { s.sync; f.labelset.dump({|dic| p.categories = dic.postln }); };
fork { s.sync; f.labelset.dump({|dic|
	//p.categories = dic.postln
	var colors = dic["data"].values.asSet.asList.collect {|v, idx| [v[0].asSymbol, idx] }.flatten.asDict;
	var colorsn = colors.size;
	var normdurs = e.collect {|v| v.duration[0] }.normalize;
	n = e.collect {|v| v.duration[0] };
	b = colors;
	dic["data"].keysValuesDo {|k,v|
		var cols = colors[v[0].asSymbol];
		var cola = (cols * 3) % colorsn / colorsn;
		var colb = (cols * 4) % colorsn / colorsn;
		var color = Color.hsv(cola, colb, 1, 0.5);
		p.pointColor_(k, color);
		p.pointSize_(k, 0.8 + (normdurs[k.split($-).last.asInteger] * 5));
		//var color = Color.hsv(colors[v[0].asSymbol] / colorsn);
		//p.pointColor_(k, color);
	};
}); };
)

(a: 3, b:4).reject {|v,k| v <  4 }
~amp = 0.1;

(
d.dump({|sdic|
	defer {
		f.labelset.dump({|dic|
			//p.categories = dic.postln
			var colors = dic["data"].values.asSet.asList.collect {|v, idx| [v[0].asSymbol, idx] }.flatten.asDict;
			var colorsn = colors.size;
			var normdurs = e.collect {|v| v.duration[0] }.normalize;
			//sdic["data"] = sdic["data"].select {|v,k| dic["data"][k][0].asSymbol == 'ZOOM0125_Tr1.WAV' }.postln;
			//p.dict_(sdic);
			defer { p = FluidPlotter(dict: sdic) };
			n = e.collect {|v| v.duration[0] };
			b = colors;
			~ps.do {|k| p.pointColor_(k, Color(0,0,0,0)) };
			~ps = [];
			//dic["data"].select {|vv,kk|vv[0].asSymbol == 'Brecht-Galiliei-Gusto-Schmankerl-Roh.wav'}
			dic["data"].select {|vv,kk|vv[0].asSymbol == dic["data"].values.asSet.asList.choose.postln }
			.keysValuesDo {|k,v|
				var cols = colors[v[0].asSymbol];
				var cola = (cols * 3) % colorsn / colorsn;
				var colb = (cols * 4) % colorsn / colorsn;
				var color = Color.hsv(cola, colb, 1, 0.6);
				p.pointColor_(k, color);
				p.pointSize_(k, 0.8 + (normdurs[k.split($-).last.asInteger] * 5));
				//var color = Color.hsv(colors[v[0].asSymbol] / colorsn);
				//p.pointColor_(k, color);
			}
		})
	}
})
)
().play
p.shape_(\circle)
p.background_(Color.black)
FluidPlotter
n.sort.plot
4 & 2r111000
12 & 2r111000
12 & 2r000111

123.asBinaryString
"01111011".asInteger
& "0000000".binaryValue


c["data"]["slice-7524"]
f.labelset.print
f.labelset.dump({|dic| c = dic })
c["data"].values.asSet
~points = {Buffer.alloc(s, 2)}!9;
~res = {Buffer.alloc(s, 35)}!9;
~lasts = nil!9;
t = {FluidKDTree(s, 1)}!9;
t.do(_.fit(d));
Ndef(\player).gui
~lasts
(
Trajectories({|index, vals|
	fork {
		~points[index].setn(0, vals[0], 1, 1-vals[1]);
		s.sync;
		t[index].kNearest(~points[index], 1, {|value|
			//index.post; " ".post; value.post; " ".post; ~lasts[index].postln;
			if (value!=~lasts[index]){
				~lasts[index] = value;
				index.post; " ".post; value.post; " ".post; ~lasts[index].postln;
				f.datasets.merged.getPoint(value, ~res[index], {
					~res[index].getn(30, 5, {|vs|
						//Ndef(\player).set(\pos, vs[1], \end, vs[2], \t_trig, 1);
						//Ndef(\player).set(\posn, vs[1]/f.buffer.numFrames, \dura, (vs[2]-vs[1])/f.buffer.sampleRate, \t_trig, 1);
							Synth(\playsl, [
						\sliceStart, vs[1], //duration[1],
						\sliceDur, (vs[2] - vs[1])/f.buffer.sampleRate, //slice.duration[0],
						\atk, 1.0.rand,
						\rls, 1.0.rand,
						\pan, 1.0.rand2, //(idx/v.size * 0.5).sin.postln,
						\amp, ~amp,
						\out, 0,
					]);
					})
				});
			};
		});
	};
})
)



(
Ndef(\player).addSpec(\pos, [0, f.buffer.numFrames]);
Ndef(\player).addSpec(\end, [0, f.buffer.numFrames]);
Ndef(\player, {arg t_trig=0, pos=0, end=0, posn=0, dura=0;
	var dur = (end - pos)/f.buffer.sampleRate;
	//PlayBuf.ar(2, f.buffer, 1, t_trig, pos, 0) * Env.new([0,1,1,0], [0,dur,0]).ar(0, t_trig);
	GrainBuf.ar(8, t_trig, dura, f.monos, \rate.kr(1, spec:[-3,3]), posn*LFNoise0.kr(5).range(0.9999,1.0001), 2, [-1,1]).sum
}).gui
)
FluidPlotter

FluidUMAP(s, 2, 15, 0.1, 400, 0.05).fitTransform(f.datasets.merged, d, {"done".postln})
FluidNormalize(s).fitTransform(d, d, {"done".postln})
d.dump({|dic| defer { FluidPlotter(dict: dic)}})

FluidMDS(s, 2).fitTransform(f.datasets.merged, d, {"done".postln})
FluidNormalize(s).fitTransform(d, d, {"done".postln})

d.dump({|dic| defer { FluidPlotter(dict: dic)}})

k = FluidKDTree(s).fit(d, {"done".postln})
d.dump({|dic| defer { p = FluidPlotter(dict: dic)}})


d.dump{|dic| dic["data"].do(_.postln)}


f.slicesFor(f.loader.files[0])

q = FluidDataSetQuery(s);
q.addRange(0, 35);
q.filter(34, "!=", 3);
r = FluidDataSet(s);
q.transform(f.datasets.merged, r);
r.print
f.datasets.merged.print



l = FluidLoadFolder("/home/bgola/workspace/samples/addtv")
l.play
l.index[l.files[0].path.basename.asSymbol][\bounds]
l.index
f.loader.index

f.datasets.duration.print

f.datasets.

f.store("/home/bgola/tmp/pankemix")

f.load("/home/bgola/tmp/pankemix")
fork { f.mergeDatasets }
f.datasets.merged.cols

d = f.export;
d.size;
FluidNormalize(s).fitTransform(f.datasets.merged, f.datasets.merged)
f.datasets.merged.print
(
SynthDef(\playsl, {arg out=0, sliceStart, sliceDur, atk=0.01, rel=0.01, amp=1, pan=0;
	Out.ar(0,
		PlayBuf.ar(2, f.buffer, 1, 1, sliceStart) * Env.new([0,1,1,0], [atk,1-(atk+rel),rel]).ar(2, 1, sliceDur)*amp
	);
}).add
)




q = FluidDataSetQuery(s);
k = FluidDataSet(s);
q.addRange(0,2);
k.print
q.transform(f.datasets.mfcc, k)
k.print
q.transformJoin(f.datasets.pitch, k, k, {"done".postln})
k.print
k.print
f.datasets.mfcc.print
f.datasets.chroma.print
transformJoin
f.datasets.all =

f.loader.files
d[10].mfcc
(
var from = 1500;
var to = 1590;
var result = [];
m = FluidDataSet(s);
c = Condition.new;
o = d[from..to];
fork {
	c.test=false;
	r=Buffer.alloc(s, 32, 1, {c.test=true;c.signal;});
	c.wait;
	c.test=false;
	m.addPoint('slice-0', r, {c.test=true;c.signal;});
	c.wait;
	c.test=false;
	m.merge(k, true, {c.test=true;c.signal;});
	c.wait;
	(from-200..to+200).do {|idx| m.deletePoint("slice-%".format(idx).asSymbol)};
	//c.test=false;c.wait;
	o.collect {|slice|
		c.test=false;
		r.loadCollection(
			slice.chroma++slice.mfcc++slice.pitch++slice.duration++[slice.loudness[0]]++slice.spectralShape,
			0, {c.test=true;c.signal;});
		c.wait;
		c.test=false;
		m.kNearest(r, 3, {|found|
			var idx = found.choose.asString.split($-).last.asInteger;
			result = result.add(d[idx]);
			c.test = true;
			c.signal;
		});
		c.wait;
	};
	r.free;
	l = result;
	result.do {|slice, idx|
		var slo;
		Synth(\playsl, [\sliceStart, slice.duration[1], \sliceDur, slice.duration[0], \atk, 0.05, \rls, 0.5, \pan, 1, \amp, 0.5]);
		slo = o[idx];
		//Synth(\playsl, [\sliceStart, slo.duration[1], \sliceDur, slice.duration[0], \atk, 0.05, \rls, 0.5, \pan, -1, \amp, 0.5]);
		(slice.duration[0]*0.95).wait;
	}
}
)


o[10].mfcc.plot
l[10].mfcc.plot
(
fork {
	o.do {|slice|
		Synth(\playsl, [\sliceStart, slice.duration[1], \sliceDur, slice.duration[0]]);
		slice.duration[0].postln.wait;
	}
}
)
l.size
o.size
(
fork {
	l.do {|slice|
		Synth(\playsl, [\sliceStart, slice.duration[1], \sliceDur, slice.duration[0]]);
		slice.duration[0].postln.wait;
	}
}
)

(
Tdef(\xx, {
	inf.do {|idx|
		Ndef(\x).set(\lpf, (idx/10).sin);
		1.0.wait;
	}
});
)

Ndef(\x, {\lpf.kr.poll})
Ndef(\x).set(\lpf, 1)
Ndef(\mod, {SinOsc.kr(1)}).scope;
Ndef(\x).set(\lpf, Ndef(\mod))

m.kNearest()
m.getPoint('slice-101', b)
m.deletePoint("slice-%".format(102).asSymbol)
f.datasets.mfcc.print
f.datasets.mfcc

d
d.size
d[1578]
d = d[..-1]
d[..d.size-2].size

d[30].chromaClass

d[30]

d.last
d=d.reject{|el| el.isEmpty}
d.size

[1,2,3,4][..2]


(asdad: 213, asdads: 3, asdadsv: 5).


[1,23,4,5].mean


d[100].mfcc.plot

d[10].keys
f.durations[10]

d.do {|el,idx|
	d[idx].duration = f.durations[idx];
}

d[0].pitch

d[0].spectralShape

(
v = e
//.select {|el| el.spectralShape[0] > -20 }
//.select {|el| el.pitch[0] < 9500.69 }
.select {|el| el.duration[0] > 0.3 and: {el.duration[0] < 0.9 } }
.select {|el| el.loudness[0] < -29 } // and: {el.loudness[0] < -10} }
//.sort {|a,b| a.pitch[0] < b.pitch[0] }
//.sort {|a,b| a.loudness[0] < b.loudness[0] }
//.sort {|a,b| a.spectralShape[0] < b.spectralShape[0]}
//.sort {|a,b| a.duration[0] < b.duration[0] }
;
v.size.postln;
)

(
SynthDef(\playsl, {arg out=0, sliceStart, sliceDur, atk=0, rel=0, amp=1, pan=0,cA=0,cB=0;
	var snd = MoogVCF.ar((PlayBuf.ar(2, f.buffer, BufRateScale.kr(f.buffer), 1, sliceStart) *
		Env.new([0,1,1,0], [atk,1-(atk+rel),rel], [cA,0,cB]).ar(0, 1, sliceDur) * amp), LFDNoise3.ar(ExpRand(1,40.0)).exprange(580,14000), 0);
	snd = snd + JPverb.ar(snd * \amp.asSpec.map(Rand(0.0, 0.1)), Rand(1.0, 5.0), Rand(0.0, 1.0), Rand(0.0, 1.0), 0.707, Rand(0.0,0.4), Rand(0.1,10));
	Out.ar(out,
		//PanAz.ar(8,
		snd * Env.new([0, 1, 1, 0], [0.01, sliceDur*Rand(1.0, 4.0), 0.01]).ar(2, 1)
		//	pan, amp, 1.2
		//)
	);
}).add
)

//~amps = \amp.asSpec.map(((0..16) / 16).reverse.sin.abs); //[1,1,1,1,1,1,1,1,1];
~amps = [1,0,0.6,1.0,0.7]
~reps = [81,3,1,8,1];

t = 1


g = g[..2]
t = 1;
g = g.reverse
g = g[..1]
g = r
r = g
g = g.sort {|a,b| a.loudness[0] < b.loudness[0] }
g = g.sort {|a,b| a.pitch[0] < b.pitch[0] }
g[0] = v.scramble.keep(1).first;//.sort {|a,b| a.loudness[0] < b.loudness[0] };

g[g.size.rand] = g.choose;
h = g
g = [g[8], g[3]]
g = g[2..]
g = v.scramble.keep(rrand(1,12)).sort {|a,b| a.pitch[0] < b.pitch[0] };
t = 0.75
b = g
g = g[3..]
g = g[..2]
(
//t =rrand(0.2,2.5).postln;

o.stop; o = fork {
	var ii = 0;
	loop {
		g.do {|slice,idx|
			var reps = ~reps[ii% ~reps.size];
			idx.postln;
			reps.do {
				Synth(\playsl, [
					\sliceStart, slice.duration[1],
					\sliceDur, slice.duration[0]/reps,
					\atk, 0.3.rand,
					\rls, 0.5.rand,
					\pan, 1.0.rand2, //(idx/v.size * 0.5).sin.postln,
					\amp, ~amps[ii % ~amps.size],
					\out, 0,
					\cA, 5.0.rand2,
					\cB, 5.0.rand2
				]);
				(slice.duration[0]*t/reps).wait;
			};
			ii = ii + 1;
			//slice.pitch[0].postln;
		}
	}
}
)

x = Buffer.read(s, "/home/bgola/workspace/CloudGarden/EchoMaterial/Sounds_hiStory/Hermann-Sherchen-Rehersal03.wav")
o = [
	"/home/bgola/workspace/CloudGarden/EchoMaterial/Sounds_hiStory/Malta-G-Nulstrahlere01.wav",
	"/home/bgola/workspace/CloudGarden/EchoMaterial/Sounds_hiStory/Malta-G-Nulstrahlere02.wav",
	"/home/bgola/workspace/CloudGarden/EchoMaterial/Sounds_hiStory/Malta-G-Nulstrahlere03.wav",
	"/home/bgola/workspace/CloudGarden/EchoMaterial/Sounds_hiStory/Malta-G-Nulstrahlere04.wav",
	"/home/bgola/workspace/CloudGarden/EchoMaterial/Sounds_hiStory/Malta-G-Nulstrahlere05.wav",
	"/home/bgola/workspace/CloudGarden/EchoMaterial/Sounds_hiStory/Malta-G-Nulstrahlere06.wav",
	"/home/bgola/workspace/CloudGarden/EchoMaterial/Sounds_hiStory/Malta-G-Nulstrahlere07-schaefer.wav",
	"/home/bgola/workspace/CloudGarden/EchoMaterial/Sounds_hiStory/Malta-G-Nulstrahlere08-schaefer.wav",
	"/home/bgola/workspace/CloudGarden/EchoMaterial/Sounds_hiStory/Malta-G-Nulstrahlere09-schaefer.wav",
	"/home/bgola/workspace/CloudGarden/EchoMaterial/Sounds_hiStory/Malta-G-Nulstrahlere10-schaefer.wav",
	"/home/bgola/workspace/CloudGarden/EchoMaterial/Sounds_hiStory/Malta-G-Nulstrahlere11-schaeferwav.wav",
	"/home/bgola/workspace/CloudGarden/EchoMaterial/Sounds_hiStory/Malta-G-Nulstrahlere13-feedback-LF.wav"
].collect {|f| Buffer.readChannel(s, f, channels: [0]); }



Ndef(\zz, {PlayBuf.ar(2, x, BufRateScale.kr(x), loop: 1).sum}).gui

(
Ndef(\gg, {
	var amp = LFDNoise3.kr(5).range(0,1);
	var pan = LFDNoise3.kr(0.2);
	var bf = Select.kr(LFDNoise1.kr(10).range(0,o.size), o);
	var pos = LFDNoise3.kr(10).range(0,1);
	GrainBufJ.ar(8, Dust.kr(3), LFDNoise3.kr(1).range(0.04,3), bf, BufRateScale.kr(bf), pos, 0, 2, amp, pan)
})
)

(
fork {
	u.do {|slice,idx|
		Synth(\playsl, [
			\sliceStart, slice.duration[1],
			\sliceDur, slice.duration[0],
			\atk, 0.1,
			\rls, 0.1,
			\pan, 0,// (idx/v.size * 2pi).sin.postln,
			\amp, 1.5,
			\out, 5,
		]);
		//idx.postln;
		//slice.pitch[0].postln;
		(slice.duration[0]*0.9).wait;
	}
}
)

v.sum{ |el| el.duration[0] } / 60

d[0].loudness

r.size


(
SynthDef(\playsl, {arg out=0, sliceStart, sliceDur, atk=0, rel=0, amp=1, pan=0;
	Out.ar(out,
		//PanAz.ar(8,
		(PlayBuf.ar(2, f.buffer, BufRateScale.kr(f.buffer), 1, sliceStart) * Env.new([0,1,1,0], [atk,1-(atk+rel),rel]).ar(2, 1, sliceDur) * amp)
		//	pan, amp, 1.2
		//)
	);
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
		//slice.pitch[0].postln;
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