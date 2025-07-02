/*
to look at:

FluidGrid
FluidHPSS

FluidKNNRegressor
FluidKNNClassifier

FluidMLPRegressor
FluidMLPClassifier

FluidSliceCorpus
FluidProcessSlices

FluidKMeans
FluidSKMeans

*/

FluidModel {
	var <buffer, <monos, <loader, <mono, <slices, <dataset, <scaler, <kdtree, <scaled_dataset, <lookup, <folder, <datasets;
	var <>slicer, <durations;
	var <s, cond;

	*new {
		^super.newCopyArgs().init;
	}

	init {
		s = Server.default;
		this.slicer = FluidModelSlicer.default();
		datasets = (mfcc: 0, chroma: 0, pitch: 0, spectralShape: 0, loudness: 0, duration: 0);
		durations = [];
	}

	loadFolder {|folder|
		var c = Condition.new;
		loader = FluidLoadFolder(folder);

		loader.play(s,{
			c.test = true;
			c.signal;
			"all files loaded".postln;
		});
		c.wait;
		^loader.buffer;
	}

	loadBothMono {
		var buff0, buff1;
		var c = Condition.new();
		buffer.postln;
		if (buffer.numChannels.postln == 2) {
			buffer.write("/tmp/multichannel.aiff", completionMessage: {|buf|
				buf.postln;
				fork {
					0.1.wait;
					"wrote, now read".postln;
					buff0 = Buffer.readChannel(s, "/tmp/multichannel.aiff", channels:[0], action: {
						buff1 = Buffer.readChannel(s, "/tmp/multichannel.aiff", channels:[1], action: {
							"loaded multichannel to mono buffers".postln;
							c.test = true;
							c.signal;
						});
					});
				};
			});
			c.wait;
		} {
			"resulting file is mono, just copy the buffer to both channels... ".post;
			buff0 = buffer;
			buff1 = buffer;
			"done".postln;
		};
		^[buff0, buff1];
	}

	toMono {
		var buf;
		var c = Condition.new();
		if(buffer.numChannels > 1){
			buf = Buffer(s);
			buffer.numChannels.do{
				arg chan_i;
				FluidBufCompose.processBlocking(s,
					buffer,
					startChan:chan_i,
					numChans:1,
					gain:buffer.numChannels.reciprocal,
					destination:buf,
					destGain:1,
					action:{
						"copied channel: %".format(chan_i).postln;
						if (chan_i == 1) {
							c.test = true;
							c.signal;
						}
					}
				);
			};
		}{
			"loader buffer is already mono".postln;
			buf = buffer;
			c.test = true;
			c.signal;
		};
		c.wait;
		^buf;
	}

	slice {
		var indices = Buffer(s);
		//FluidBufOnsetSlice.processBlocking(s,~source_buf,indices:~source_indices_buf,metric:9,threshold:0.2,minSliceLength:9,action:{
		var c = Condition.new();
		var action = {
			indices.loadToFloatArray(action:{ arg indices_array;
				"found % slices".format(indices_array.size-1).postln;
				"average length: % seconds".format((buffer.duration / (indices_array.size-1)).round(0.001)).postln;
				c.test = true;
				c.signal;
			});
		};

		"Slicing...".postln;
		//FluidBufOnsetSlice.processBlocking(s,buffer,indices:indices,metric:9,threshold:0.5,minSliceLength:9,action:{
		//FluidBufNoveltySlice.processBlocking(s,buffer,indices:indices, algorithm: 1,threshold:0.25,kernelSize: 15, minSliceLength:3,action:{
		//	indices.loadToFloatArray(action:action)
		//});

		this.slicer.slice(mono, indices, action: action);
		c.wait;
		^indices;
	}


	analyze {
		var features_buf = {Buffer(s)}!5;
		var stats_buf = {Buffer(s)}!5;
		var flat_buf = {Buffer(s)}!5;
		var dur_buf = Buffer.alloc(s, 3);
		var duration = [];

		var mfccDS = FluidDataSet(s);
		var chromaDS = FluidDataSet(s);
		var pitchDS = FluidDataSet(s);
		var spectralshapeDS = FluidDataSet(s);
		var loudnessDS = FluidDataSet(s);
		var durationDS = FluidDataSet(s);

		var c = Condition.new();
		var datasets = (mfcc: mfccDS, chroma: chromaDS, pitch: pitchDS, spectralShape: spectralshapeDS,
			loudness: loudnessDS, duration: durationDS);

		"Starting analysis".postln;

		slices.loadToFloatArray(action: { arg slices_array;
			fork {
				// iterate over each index in this array, paired with this next neighbor so that we know where to start
				// and stop the analysis
				slices_array.doAdjacentPairs{
					arg start_frame, end_frame, slice_index;
					var num_frames = end_frame - start_frame;
					"analyzing slice: % / %".format(slice_index + 1,slices_array.size - 1).postln;
					FluidBufMFCC.processBlocking(s,mono,start_frame,num_frames,features: features_buf[0],startCoeff:1,numCoeffs:13);
					FluidBufStats.processBlocking(s,features_buf[0],stats:stats_buf[0],select:[\mean]);
					FluidBufFlatten.processBlocking(s,stats_buf[0],destination:flat_buf[0]);
					mfccDS.addPoint("slice-%".format(slice_index),flat_buf[0]);

					FluidBufChroma.processBlocking(s,mono,start_frame,num_frames,features:features_buf[1],normalize:1);
					FluidBufStats.processBlocking(s,features_buf[1],stats:stats_buf[1],select:[\mean]);
					FluidBufFlatten.processBlocking(s,stats_buf[1],destination:flat_buf[1]);
					chromaDS.addPoint("slice-%".format(slice_index),flat_buf[1]);

					FluidBufPitch.processBlocking(s,mono,start_frame,num_frames,features:features_buf[2],select:[\pitch, \confidence],algorithm:2);
					FluidBufStats.processBlocking(s,features_buf[2],stats:stats_buf[2],select:[\mean]);
					FluidBufFlatten.processBlocking(s,stats_buf[2],destination:flat_buf[2]);
					pitchDS.addPoint("slice-%".format(slice_index),flat_buf[2]);

					FluidBufSpectralShape.processBlocking(s,mono,start_frame,num_frames,features:features_buf[3], select:[\flatness]);
					FluidBufStats.processBlocking(s,features_buf[3],stats:stats_buf[3],select:[\mean]);
					FluidBufFlatten.processBlocking(s,stats_buf[3],destination:flat_buf[3]);
					spectralshapeDS.addPoint("slice-%".format(slice_index),flat_buf[3]);

					FluidBufLoudness.processBlocking(s,mono,start_frame,num_frames,features:features_buf[4]);
					FluidBufStats.processBlocking(s,features_buf[4],stats:stats_buf[4],select:[\mean]);
					FluidBufFlatten.processBlocking(s,stats_buf[4],destination:flat_buf[4]);
					loudnessDS.addPoint("slice-%".format(slice_index),flat_buf[4]);

					dur_buf.set(0, num_frames/mono.sampleRate);
					dur_buf.set(1, start_frame);
					dur_buf.set(2, end_frame);
					s.sync;
					durationDS.addPoint("slice-%".format(slice_index),dur_buf);
					duration = [num_frames/mono.sampleRate, start_frame, end_frame];
					durations = durations.add(duration);
					s.sync;
					//if((slice_index % 100) == 99){s.sync};
				};
				s.sync;
				"Done with analysis".postln;
				c.test = true;
				c.signal;
			};
		});
		c.wait;
		^datasets;
	}


	export {
		var nslices = slices.numFrames;
		var slices_export = nil!nslices;
		fork {
			var dumps=();
			var c = Condition.new;
			datasets.keys.do {|key,idx|
				c.test=false;
				"exporting %: %/%".format(key, idx+1, datasets.size).postln;
				datasets[key].dump {arg data;
					"   copying data...".post;
					if (data["data"].isEmpty.not) {
						dumps[key] = data["data"];
					};
					"done".postln;
					c.test=true;
					c.signal;
				};
				c.wait;
			};

			nslices.do {arg sliceidx;
				var key = "slice-%".format(sliceidx);
				var slice_data = ();
				dumps.keys.do {|dkey|
					slice_data[dkey] = dumps[dkey][key];
				};
				slice_data['chromaClass'] = slice_data.chroma.maxIndex;
				slices_export[sliceidx] = slice_data;
			};
		};
		^slices_export;
	}

	reduce {}

	fitKDTree {
		var kdtree = FluidKDTree(s);
		var scaled_dataset = FluidDataSet(s);
		var scaler;
		var c = Condition.new();

		//scaler = FluidStandardize(s);
		scaler = FluidNormalize(s, -1, 1);
		//scaler = FluidRobustScale(s);

		"Fitting KDTree".postln;
		scaler.fitTransform(datasets.mfcc,scaled_dataset, {
			"scaled...".postln;
			kdtree.fit(scaled_dataset,{
				"kdtree is fit".postln;
				c.test = true;
				c.signal;
			});
		});
		c.wait;
		^[scaler, kdtree, scaled_dataset];
	}

	makeLookup {
		var lookup = FluidDataSet(s);
		var c = Condition.new();
		scaled_dataset.size({|size|
			var dic = Dictionary.new;
			var data = size.collect {|i|["slice-%".format(i), i]}.flatten;
			dic.put(\cols, 1);
			dic.put(\data, Dictionary.newFrom(data));
			lookup.load(dic, {
				"Lookup is ready".postln;
				c.test = true;
				c.signal;
			});
		});
		c.wait;
		^lookup;
	}

	build { arg path, action;
		var scaler_tree;
		folder = path;
		fork {
			buffer = this.loadFolder(folder);
			monos = this.loadBothMono;
			mono = this.toMono;
			slices = this.slice;
			this.analyze.keysValuesDo {|key, value|
				datasets[key] = value;
			};

			scaler_tree = this.fitKDTree;

			scaler = scaler_tree[0];
			kdtree = scaler_tree[1];
			scaled_dataset = scaler_tree[2];
			lookup = this.makeLookup;
			if (action.notNil) {
				action.(this);
			};
		}
	}

	store { arg path;
		var model = this;
		var name = path.basename;
		path.mkdir;
		model.buffer.write(path +/+ "%.aiff".format(name));
		model.mono.write(path +/+ "%_mono.aiff".format(name));
		model.slices.write(path +/+ "%_slices.aiff".format(name), "aiff", "float");
		model.datasets.keys.do {|key|
			model.datasets[key].write(path +/+ "%_%_dataset.json".format(key, name));
		};
		model.scaler.write(path +/+  "%_scaler.json".format(name));
		model.kdtree.write(path +/+  "%_kdtree.json".format(name));
		model.scaled_dataset.write(path +/+  "%_scaled_dataset.json".format(name));
		model.lookup.write(path +/+  "%_lookup.json".format(name));
	}

	load { arg path;
		var name = path.basename;
		buffer = Buffer.read(s, path +/+ "%.aiff".format(name));
		monos = [nil,nil];
		monos[0] = Buffer.readChannel(s, path +/+ "%.aiff".format(name), channels:[0]);
		monos[1] = Buffer.readChannel(s, path +/+ "%.aiff".format(name), channels:[1]);
		mono = Buffer.read(s, path +/+ "%_mono.aiff".format(name));
		slices = Buffer.read(s, path +/+ "%_slices.aiff".format(name));
		datasets.keys.do {|key|
			datasets[key] = FluidDataSet(s).read(path +/+ "%_%_dataset.json".format(key, name));
		};
		scaler = FluidNormalize(s, -1, 1).read(path +/+  "%_scaler.json".format(name));
		kdtree = FluidKDTree(s).read(path +/+  "%_kdtree.json".format(name));
		scaled_dataset = FluidDataSet(s).read(path +/+  "%_scaled_dataset.json".format(name));
		lookup = FluidDataSet(s).read(path +/+  "%_lookup.json".format(name));
	}
}


FluidModelSlicer {
	var slicerClass, slicerArguments;

	*new { arg slicerClass, slicerArguments;
		^super.newCopyArgs(slicerClass, slicerArguments);
	}

	*default {
		^this.novelty();
	}

	*novelty { arg algorithm=0, kernelSize=3, threshold=0.5, filterSize=1,
		minSliceLength=2, windowSize=1024, hopSize= -1, fftSize= -1;
		^this.new(
			FluidBufNoveltySlice,
			[algorithm, kernelSize, threshold, filterSize, minSliceLength, windowSize, hopSize, fftSize]
		);
	}

	*onset { arg metric=0, threshold=0.5, minSliceLength=2, filterSize=5, frameDelta=0,
		windowSize=1024, hopSize= -1, fftSize= -1;
		^this.new(
			FluidBufOnsetSlice,
			[metric, threshold, minSliceLength, filterSize, frameDelta, windowSize, hopSize, fftSize]
		);
	}

	*amp { arg fastRampUp=1, fastRampDown=1, slowRampUp=100, slowRampDown=100, onThreshold= -144,
		offThreshold= -144, floor= -144, minSliceLength=2, highPassFreq=85;
		^this.new(
			FluidBufAmpSlice,
			[fastRampUp, fastRampDown, slowRampUp, slowRampDown, onThreshold, offThreshold, floor, minSliceLength, highPassFreq]
		);
	}

	*transient { arg order=20, blockSize=256, padSize=128, skew=0, threshFwd=2, threshBack=1.1, windowSize=14,
		clumpLength=25, minSliceLength=1000;
		^this.new(
			FluidBufTransientSlice,
			[order, blockSize, padSize, skew, threshFwd, threshBack, windowSize, clumpLength, minSliceLength]
		);
	}

	slice { arg source, indices, action;
		// args is: [server, source, startFrame, numFrames, startChan, numChans, SLICER ARGS, freeWhenDone, action]
		var args = [Server.default, source, 0, -1, 0, -1, indices] ++ slicerArguments ++ [true, action];
		^slicerClass.process(*args);
	}
}


FluidModelAnalyser {

}
