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
	var <buffer, <monos, <loader, <mono, <slices, <dataset, <scaler, <kdtree;
	var <scaled_dataset, <lookup, <folder, <datasets, <labelset;
	var <>slicer;
	var <s, cond;

	*new {
		^super.newCopyArgs().init;
	}

	init {
		s = Server.default;
		this.slicer = FluidModelSlicer.default();
		datasets = (mfcc: 0, chroma: 0, pitch: 0, spectralShape: 0, loudness: 0, duration: 0, merged: 0);
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
		this.slicer.slice(mono, indices, action: action);
		c.wait;
		^indices;
	}

	chainFlatten { arg buf, ds, slice_index;
		var stats = Buffer(s);
		var flat = Buffer(s);
		FluidBufStats.processBlocking(s, buf, stats:stats,select:[\mean]);
		FluidBufFlatten.processBlocking(s, stats, destination: flat);
		ds.addPoint("slice-%".format(slice_index), flat);
		stats.free; flat.free;
	}

	chainMFCC { arg buf, start_frame, num_frames, slice_index, dss;
		FluidBufMFCC.processBlocking(s,mono,start_frame,num_frames,features: buf, startCoeff:1,numCoeffs:13);
		this.chainFlatten(buf, dss[\mfcc], slice_index);
	}

	chainChroma { arg buf, start_frame, num_frames, slice_index, dss;
		FluidBufChroma.processBlocking(s,mono,start_frame,num_frames,features: buf);
		this.chainFlatten(buf, dss[\chroma], slice_index);
	}

	chainPitch { arg buf, start_frame, num_frames, slice_index, dss;
		FluidBufPitch.processBlocking(s,mono,start_frame,num_frames,features: buf, select: [\pitch, \confidence]);
		this.chainFlatten(buf, dss[\pitch], slice_index);
	}

	chainShape { arg buf, start_frame, num_frames, slice_index, dss;
		FluidBufSpectralShape.processBlocking(s,mono,start_frame,num_frames,features: buf);
		this.chainFlatten(buf, dss[\spectralShape], slice_index);
	}

	chainLoudness { arg buf, start_frame, num_frames, slice_index, dss;
		FluidBufLoudness.processBlocking(s,mono,start_frame,num_frames,features: buf);
		this.chainFlatten(buf, dss[\loudness], slice_index);
	}

	chainDuration { arg buf, start_frame, end_frame, num_frames, slice_index, current_file, dss;
		var duration_values = [
			num_frames/mono.sampleRate,
			start_frame,
			end_frame,
			slice_index,
			current_file
		];
		buf.setn(0, duration_values);
		s.sync;
		dss[\duration].addPoint("slice-%".format(slice_index), buf);
	}

	analyze { arg num_threads;
		var mfccDS = FluidDataSet(s);
		var chromaDS = FluidDataSet(s);
		var pitchDS = FluidDataSet(s);
		var spectralshapeDS = FluidDataSet(s);
		var loudnessDS = FluidDataSet(s);
		var durationDS = FluidDataSet(s);

		var datasets = (mfcc: mfccDS, chroma: chromaDS, pitch: pitchDS, spectralShape: spectralshapeDS,
			loudness: loudnessDS, duration: durationDS);
		var buffers = {(mfcc: Buffer(s), chroma: Buffer(s), pitch: Buffer(s), specShape: Buffer(s), loudness: Buffer(s), duration: Buffer.alloc(s, 5))}!num_threads;
		var mainCond = Condition.new;
		var slices_array, threads_slices = ();
		var threads = [];

		labelset = FluidLabelSet(s);

		"Starting analysis".postln;
		slices.loadToFloatArray(action: { arg array;
			slices_array = array;
			mainCond.test = true; mainCond.signal;
		});
		mainCond.wait; mainCond.test = false;
		s.sync;

		slices_array.doAdjacentPairs { arg start_frame, end_frame, slice_index;
			var num_frames, dur_buf, current_file;
			var thread = slice_index % num_threads;

			threads_slices[thread] = threads_slices[thread].add([start_frame, end_frame, slice_index]);
		};

		threads = threads_slices.collect { |thread_slices, thread|
			fork {
				thread_slices.do {|slice|
					var start_frame = slice[0];
					var end_frame = slice[1];
					var slice_index = slice[2];
					var num_frames, dur_buf;

					var found = false;
					var current_file_idx = 0;
					var current_file = loader.files[current_file_idx];
					num_frames = end_frame - start_frame;

					while { current_file.notNil and: { found.not } } {
						var current = current_file.path.basename.asSymbol;
						if (start_frame < loader.index[current][\bounds][1]) {
							found = true;
						};
						if (found.not) {
							current_file_idx = current_file_idx + 1;
							current_file = loader.files[current_file_idx];
						};
					};

					"analyzing slice: % / %".format(slice_index + 1,slices_array.size - 1).postln;

					this.chainMFCC(buffers[thread].mfcc, start_frame, num_frames, slice_index, datasets);
					this.chainChroma(buffers[thread].chroma, start_frame, num_frames, slice_index, datasets);
					this.chainPitch(buffers[thread].pitch, start_frame, num_frames, slice_index, datasets);
					this.chainShape(buffers[thread].specShape, start_frame, num_frames, slice_index, datasets);
					this.chainLoudness(buffers[thread].loudness, start_frame, num_frames, slice_index, datasets);
					this.chainDuration(buffers[thread].duration, start_frame, end_frame, num_frames, slice_index,
						current_file_idx, datasets);

					labelset.addLabel("slice-%".format(slice_index), current_file.path.basename.asSymbol);

					s.sync;
					datasets[\duration].size({|size|
						if (size >= (slices_array.size - 1)) {
							mainCond.test = true; mainCond.signal;
						}
					});

				}
			}
		};

		mainCond.wait;
		threads.do(_.stop);

		"Done with analysis".postln;

		^datasets;
	}

	mergeDatasets {
		var query = FluidDataSetQuery(s);
		var c = Condition.new;
		var normalize, keys, cols, ds;
		keys = datasets.skeys.reject {|ds| ds == \merged }.asArray;
		datasets.merged = FluidDataSet(s);
		ds = keys[0];
		c.test = false;

		datasets[ds].cols({|ncols| cols = ncols; c.test=true; c.signal });
		c.wait; c.test = false;

		query.addRange(0, (cols), {c.test=true; c.signal});
		c.wait; c.test = false;

		query.transform(datasets[ds], datasets.merged, {c.test=true; c.signal});
		c.wait; c.test = false;

		keys[1..].do {|ds|
			query.free;
			query = FluidDataSetQuery(s);
			c.test=false;
			datasets[ds].cols({|ncols| cols = ncols; c.test=true; c.signal });
			c.wait; c.test = false;
			query.addRange(0, (cols), {c.test=true; c.signal});
			c.wait; c.test = false;
			query.transformJoin(datasets[ds], datasets.merged, datasets.merged, {c.test=true; c.signal});
			c.wait; c.test = false;
		};
	}

	// TODO: make this more flexible so we can include the window in other windows
	plot { arg dataset, action;
		var plotds = FluidDataSet(s);
		var tree, point, return, plotter, treeAction;
		var ret = (plotter: nil, action: nil, pcanorm: nil, norm: nil, pca: nil, tree: nil);

		point = Buffer.alloc(s, 2);
		return = Buffer(s);

		if (dataset.isNil) {
			dataset = datasets[\merged];
		};

		if (dataset.isKindOf(Symbol)) {
			dataset = datasets[dataset];
		};

		treeAction = { arg plotter, x, y, mod, btNum, clickCnt;
			var point = Buffer.alloc(s, 2);
			fork {
				s.sync;
				point.setn(0, x, 1, y);
				tree.kNearest(point, 1, {|value|
					datasets.duration.getPoint(value, return, {
						if (action.notNil) {
							action.(value, return);
						} {
							return.getn(0, 5, {|vals|
								var file = loader.files[vals.last.asInteger];
								var bounds = loader.index[file.path.basename.asSymbol][\bounds];

								"Slice from file % starting at second % lasting for % seconds".format(
									file.path,
									(vals[1] - bounds[0]) / mono.sampleRate,
									vals[0]).postln;
							})
						};
					})
				});
				if (point.numFrames.notNil) {
					point.free;
				}
			};
		};



		FluidNormalize(s).fitTransform(dataset, plotds);
		FluidPCA(s, 2).fitTransform(plotds, plotds);
		ret[\pcanorm] = FluidNormalize(s).fitTransform(plotds, plotds);

		tree = FluidKDTree(s, 1);
		tree.fit(plotds);
		ret[\tree] = tree;

		plotds.dump({|dic| defer {
			plotter = FluidPlotter(dict: dic, mouseMoveAction: {|plotter, x,y, mod, btNum, clickCnt|
				treeAction.(plotter, x, y, mod, btNum, clickCnt);
			});
			plotter.userView.mouseUpAction_({});
			ret[\plotter] = plotter;
			ret[\action] = { |ret, x, y| treeAction.(ret.plotter, x, y)  };
		}});

		fork {
			s.sync;
			labelset.dump({|dic|
				var colors = dic["data"].values.asSet.asList.collect {|v, idx| [v[0].asSymbol, idx] }.flatten.asDict;
				var colorsn = colors.size;
				dic["data"].keysValuesDo {|k,v|
					var cols = colors[v[0].asSymbol];
					var cola = (cols * 3) % colorsn / colorsn;
					var colb = (cols * 4) % colorsn / colorsn;
					var colc = (cols * 5) % colorsn / colorsn;
					var color = Color.hsv(cola, colb.linlin(0,1,0.5,1.0), colc.linlin(0,1,0.5,1), 0.7);
					plotter.pointColor_(k, color);
				};
			});
		};

		^ret;
	}

	export {
		var nslices = slices.numFrames-1;
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
				if (slice_data[\chroma].notNil) {
					slice_data['chromaClass'] = slice_data.chroma.maxIndex;
				};
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
		kdtree.fit(datasets.merged,{
			"kdtree is fit".postln;
			c.test = true;
			c.signal;
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

	build { arg path, num_threads=1, action;
		var scaler_tree;
		folder = path;
		fork {
			buffer = this.loadFolder(folder);
			monos = this.loadBothMono;
			mono = this.toMono;
			slices = this.slice;
			this.analyze(num_threads).keysValuesDo {|key, value|
				datasets[key] = value;
			};
			this.mergeDatasets;

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
		model.labelset.write(path +/+ "%_labelset.json".format(name));
		model.scaler.write(path +/+  "%_scaler.json".format(name));
		model.kdtree.write(path +/+  "%_kdtree.json".format(name));
		model.scaled_dataset.write(path +/+  "%_scaled_dataset.json".format(name));
		model.lookup.write(path +/+  "%_lookup.json".format(name));
	}

	load { arg path, loadDatasets=true;
		var name = path.basename;
		buffer = Buffer.read(s, path +/+ "%.aiff".format(name));
		monos = [nil,nil];
		monos[0] = Buffer.readChannel(s, path +/+ "%.aiff".format(name), channels:[0]);
		monos[1] = Buffer.readChannel(s, path +/+ "%.aiff".format(name), channels:[1]);
		mono = Buffer.read(s, path +/+ "%_mono.aiff".format(name));
		slices = Buffer.read(s, path +/+ "%_slices.aiff".format(name));
		if (loadDatasets) {
			datasets.keys.do {|key|
				"loading %...".format(key).postln;
				datasets[key] = FluidDataSet(s).read(path +/+ "%_%_dataset.json".format(key, name));
			};
			scaler = FluidNormalize(s, -1, 1).read(path +/+  "%_scaler.json".format(name));
			kdtree = FluidKDTree(s).read(path +/+  "%_kdtree.json".format(name));
			scaled_dataset = FluidDataSet(s).read(path +/+  "%_scaled_dataset.json".format(name));
			lookup = FluidDataSet(s).read(path +/+  "%_lookup.json".format(name));
			labelset = FluidLabelSet(s).read(path +/+ "%_labelset.json".format(name));
		};
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

	*novelty { arg algorithm=0, kernelSize=15, threshold=0.25, filterSize=1,
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


+ FluidPlotter {

	/*
    *createCatColors {
		// colors from: https://github.com/d3/d3-scale-chromatic/blob/main/src/categorical/category10.js
		^100.collect {|idx| Color.hsv(idx/100, 1, 1) };
	}
    */

	categories_ {
		arg labelSetDict;

		if(dict_internal.size != 0,{
			var label_to_int = Dictionary.new;
			var counter = 0;
			dict_internal.keysValuesDo({
				arg id, fp_pt;

				// the id has to be converted back into a string because the
				// labelSetDict that comes in has the keys as strings by default
				var category_string = labelSetDict.at("data").at(id.asString)[0];
				var category_int;
				var color;

				if(label_to_int.at(category_string).isNil,{
					label_to_int.put(category_string,counter);
					counter = counter + 1;
				});

				category_int = label_to_int.at(category_string);

				/*if(category_int > (categoryColors.size-1),{
					"FluidPlotter:setCategories_ FluidPlotter doesn't have that many category colors. You can use the method 'setColor_' to set colors for individual points.".warn
				});

				color = categoryColors[category_int];
				fp_pt.color_(color);*/
			});
			this.refresh;
		},{
			"FluidPlotter::setCategories_ FluidPlotter cannot receive method \"categories_\". It has no data. First set a dictionary.".warn;
		});
	}
}

