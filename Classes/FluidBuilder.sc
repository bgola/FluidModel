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
	classvar <all;
	classvar <queryFeatures;

	var <key, <buffer, <monos, <loader, <mono, <slices, <scaler, <kdtree;
	var <scaled_dataset, <lookup, <datasets, <labelset, <datasets_info;
	var <>slicer, <slice_cache;
	var <s, <server, cond;
	var <nmf_bases;

	*initClass {
		all = ();
		queryFeatures = (
			spectralShape: [\centroid, \spread, \skewness, \kurtosis, \rolloff, \flatness, \crest],
			pitch: [\pitch, \confidence],
			duration: [\duration],
			mfcc: 13.collect {|i| "mfcc_%".format(i).asSymbol },
			loudness: [\loudness, \peak],
			chroma: 12.collect {|i| "chroma_%".format(i).asSymbol },
			slice_info: [\frames, \start, \end, \slice_index, \file],
		);
		queryFeatures[\merged] = queryFeatures.skeys.collect { |k| queryFeatures[k] }.flatten;
	}

	*new { arg key;
		var model = all[key];
		if (model.isNil) {
			model = super.newCopyArgs(key).init;
		};
		all[key] = model;
		^model;
	}

	init {
		server = Server.default;
		s = server;
		this.slicer = FluidModelSlicer.default();
		labelset = FluidLabelSet(s);
		slice_cache = ();
		datasets = (mfcc: 0, chroma: 0, pitch: 0, spectralShape: 0, loudness: 0, duration: 0, slice_info: 0, merged: 0);
		datasets_info = ();
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

	loadMultiple {|files|
		var c = Condition.new;
		loader = FluidLoadMultiple(files);

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
		ds.addPoint("%-%".format(key, slice_index), flat);
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

	chainDuration { arg buf, num_frames, slice_index, dss;
		var duration_values = [
			num_frames/mono.sampleRate,
		];
		buf.setn(0, duration_values);
		s.sync;
		dss[\duration].addPoint("%-%".format(key, slice_index), buf);
	}

	chainSliceInfo { arg buf, start_frame, end_frame, num_frames, slice_index, current_file, dss;
		var slice_info_values = [
			num_frames,
			start_frame,
			end_frame,
			slice_index,
			current_file,
		];
		buf.setn(0, slice_info_values);
		slice_cache["%-%".format(key, slice_index).asSymbol] = slice_info_values;
		s.sync;
		dss[\slice_info].addPoint("%-%".format(key, slice_index), buf);
	}

	calculateNMF { arg sample_duration, nbases, action;
		this.sampleCorpus(sample_duration, {|buf|
			nmf_bases = Buffer(s);
			FluidBufNMF.processBlocking(s, buf, 0, -1, 0, -1, -1, 0, nmf_bases, 0, components: nbases, iterations: 200, action: action);
		});
	}

	analyze { arg num_threads;
		var mfccDS = FluidDataSet(s);
		var chromaDS = FluidDataSet(s);
		var pitchDS = FluidDataSet(s);
		var spectralshapeDS = FluidDataSet(s);
		var loudnessDS = FluidDataSet(s);
		var durationDS = FluidDataSet(s);
		var slice_infoDS = FluidDataSet(s);

		var datasets = (mfcc: mfccDS, chroma: chromaDS, pitch: pitchDS, spectralShape: spectralshapeDS,
			loudness: loudnessDS, duration: durationDS, slice_info: slice_infoDS);
		var buffers = {
			(mfcc: Buffer(s), chroma: Buffer(s), pitch: Buffer(s), specShape: Buffer(s), loudness: Buffer(s),
				duration: Buffer.alloc(s, 1), slice_info: Buffer.alloc(s, 5))
		}!num_threads;
		var mainCond = Condition.new;
		var slices_array, threads_slices = ();
		var threads = nil!num_threads;

		labelset = FluidLabelSet(s);

		"Starting analysis".postln;
		slices.loadToFloatArray(action: { arg array;
			slices_array = array;
			mainCond.test = true; mainCond.signal;
		});
		mainCond.wait; mainCond.test = false;

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
					this.chainDuration(buffers[thread].duration, num_frames, slice_index, datasets);
					this.chainSliceInfo(buffers[thread].slice_info, start_frame, end_frame, num_frames, slice_index, current_file_idx, datasets);

					labelset.addLabel("%-%".format(key, slice_index), current_file.path.basename.asSymbol);
					s.sync;

					datasets[\slice_info].size({|size|
						if (size >= (slices_array.size - 1)) {
							mainCond.test = true; mainCond.signal;
						}
					});
				}
			}
		};

		mainCond.wait;
		"stopping threads".postln;
		threads.do(_.stop);

		"Done with analysis".postln;

		^datasets;
	}

	mergeDatasets { arg keys=#[];
		var query = FluidDataSetQuery(s);
		var c = Condition.new;
		var normalize, cols, ds;
		var merged;
		if (keys.isEmpty) {
			keys = datasets.skeys.reject {|ds| ds == \merged }.asArray;
		};
		keys = keys.sort;
		merged = FluidDataSet(s);
		ds = keys[0];
		c.test = false;

		datasets[ds].cols({|ncols| cols = ncols; c.test=true; c.signal });
		c.wait; c.test = false;

		query.addRange(0, (cols), {c.test=true; c.signal});
		c.wait; c.test = false;

		query.transform(datasets[ds], merged, {c.test=true; c.signal});
		c.wait; c.test = false;

		keys[1..].do {|ds|
			query.free;
			query = FluidDataSetQuery(s);
			c.test=false;
			datasets[ds].cols({|ncols| cols = ncols; c.test=true; c.signal });
			c.wait; c.test = false;
			query.addRange(0, (cols), {c.test=true; c.signal});
			c.wait; c.test = false;
			query.transformJoin(datasets[ds], merged, merged, {c.test=true; c.signal});
			c.wait; c.test = false;
		};
		^merged;
	}

	prReduceDataset { arg dataset, n, algo, grid, action;
		var reduxds;
		var tree, return, treeAction;
		var ret = (norm: nil, algo: nil, tree: nil, dataset: nil, find: nil);

		reduxds = FluidDataSet(s);
		ret[\dataset] = reduxds;
		return = Buffer(s);

		FluidNormalize(s).fitTransform(dataset, reduxds);

		if (algo == \pca) {
			ret[\algo] = FluidPCA(s, n);
		};

		if (algo == \umap) {
			ret[\algo] = FluidUMAP(s, n, 5, iterations:1300);
		};

		if (algo == \mds) {
			ret[\algo] = FluidMDS(s, n, 0);
		};

		ret[\algo].fitTransform(reduxds, reduxds);

		ret[\norm] = FluidNormalize(s);
		ret[\norm].fitTransform(reduxds, reduxds);

		if (grid > 0) {
			if (n != 2) {
				"FluidModel: grid can only be used for 2 dimensions".warn;
			} {
				FluidGrid(s, grid).fitTransform(reduxds, reduxds);
				FluidNormalize(s).fitTransform(reduxds, reduxds);
			};
		};


		tree = FluidKDTree(s, 1);
		tree.fit(reduxds);
		ret[\tree] = tree;

		treeAction = { arg coordinates, action;
			var point = Buffer.alloc(s, n);
			coordinates.postln;
			fork {
				s.sync;
				point.setn(0, coordinates);
				tree.kNearest(point, 1, {|slice|
					var modelkey = slice.asString.split($-).drop(-1).join($-).asSymbol;

					// this is in case this is being called by a ModelCompose
					var model = FluidModel(modelkey);
					var slice_info = slice_cache[slice.asSymbol];

					if (action.notNil) {
						action.(model, slice);
					} {
						var file = model.loader.files[slice_info[4].asInteger];
						var bounds = model.loader.index[file.path.basename.asSymbol][\bounds];

						"Slice from file % starting at second % lasting for % seconds".format(
							file.path,
							(slice_info[1] - bounds[0]) / model.mono.sampleRate,
							slice_info[0] / model.mono.sampleRate).postln;
					};
				});

				if (point.numFrames.notNil) {
					point.free;
				}
			};
		};

		ret[\find] = { arg self, coordinates, findAction; treeAction.(coordinates, findAction); self };

		action.value(ret.postln);
		^ret;
	}

	reduceDimension { arg n=2, algo=\pca, dataset, grid=0, action;
		var ret = ();

		if (dataset.isNil) {
			dataset = datasets[\merged];
		};

		if (dataset.isKindOf(Symbol)) {
			dataset = datasets[dataset];
		};

		forkIfNeeded {
			if (dataset.isCollection) {
				dataset = this.mergeDatasets(dataset);
			};

			if (dataset.isKindOf(FluidDataSet)) {
				this.prReduceDataset(dataset, n, algo, grid, action).keysValuesDo {|k,v|
					ret[k] = v;
				};
			};
		};
		^ret;
	}

	// TODO: make this more flexible so we can include the window in other windows
	plot { arg dataset, algo=\pca, grid=0, action;
		var plotter;
		var ret = (plotter: nil, redux: nil);

		ret[\redux] = this.reduceDimension(2, algo, dataset, grid, { arg redux;

			redux.dataset.dump({|dic| defer {
				plotter = FluidPlotter(dict: dic, mouseMoveAction: {|plotter, x,y, mod, btNum, clickCnt|
					redux.find([x,y], action);
				});
				plotter.userView.mouseUpAction_({});
				ret[\plotter] = plotter;

                 forkIfNeeded {
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
			}});
		});

		^ret;
	}

	filter { arg queries=#[], srcDataset, action;
		var fquery = FluidDataSetQuery(s);
		var ds = FluidDataSet(s);
		var columns;
		var col, condition, value;
		srcDataset = srcDataset ? datasets[\merged];
		columns = FluidModel.queryFeatures[\merged];
		fquery.addRange(0, columns.size);

		# col, condition, value = queries[0].split($ );
		col = columns.indexOf(col.asSymbol);
		fquery.filter(col, condition.asSymbol, value.asNumberIfPossible);
		queries[1..].do {|query|
			var or;
			# or, col, condition, value = query.split($ );
			col = columns.indexOf(col.asSymbol);
			if (or.toLower == "or") {
				fquery.or(col, condition.asSymbol, value.asNumberIfPossible);
			} {
				fquery.and(col, condition.asSymbol, value.asNumberIfPossible);
			}
		};
		fquery.transform(srcDataset, ds, action);

		^ds;
	}

	subset { arg ds, keys, action;
		var fquery = FluidDataSetQuery(s);
		var rds = FluidDataSet(s);

		keys.sort.collect {|k|
			FluidModel.queryFeatures[k].collect {|col|
				fquery.addColumn(FluidModel.queryFeatures[\merged].indexOf(col))
			};
		}.flat;

		fquery.transform(ds, rds, action);

		^rds;
	}

	sort { arg ds, feature, order=\asc, shape, action;
		var ret = (result: nil);
		if (shape.isNil) {
			shape = \merged;
		};
		if (shape.isKindOf(Symbol)) {
			shape = [shape];
		};

		shape = shape.sort.collect {|k| FluidModel.queryFeatures[k] }.flat;

		ds.dump({|dic|
			ret.result = [];
			dic["data"].keysValuesDo {|k,v|
				var obj = ();
				obj.key = k;
				shape.collect {|feat, idx|
					obj[feat] = v[idx]
				};
				ret.result = ret.result.add(obj);
			};
			if (order == \asc) {
				ret.result = ret.result.sort({|a,b| a[feature] < b[feature]});
			} {
				ret.result = ret.result.sort({|a,b| a[feature] > b[feature]});
			};
			action.(ret.result);
		});
		^ret;
	}

	sampleCorpus { arg duration, action;
		var durationInSamples = this.mono.sampleRate * duration;
		var totalSamples = this.mono.numFrames;
		var percentage = durationInSamples / totalSamples;
		var counter = 0;
		var buffer = Buffer.alloc(s,durationInSamples,1);
		forkIfNeeded {
			var frame_sum = 0;
			s.sync;
			buffer.updateInfo;
			s.sync;
			this.loader.files.do {|file,i|
				var identifier = file.path.basename.asSymbol;
				var numframes = file.numFrames * percentage;
				var startframe = this.loader.index[identifier][\bounds][0];
				var sampleStartFrame = (startframe + file.numFrames/2) - (numframes/2);

				this.mono.copyData(buffer, frame_sum, sampleStartFrame, numframes);
				s.sync;
				buffer.updateInfo;
				s.sync;
				counter = counter + 1;
				frame_sum = frame_sum + numframes;
				if(counter == (this.loader.files.size)) {action !? action.value(buffer)};
			};
		};
		^buffer;
	}

	export {
		var nslices = slices.numFrames-1;
		var slices_export = nil!nslices;
		fork {
			var dumps=();
			var c = Condition.new;
			datasets.keys.do {|k,idx|
				c.test=false;
				"exporting %: %/%".format(k, idx+1, datasets.size).postln;
				datasets[k].dump {arg data;
					"   copying data...".post;
					if (data["data"].isEmpty.not) {
						dumps[k] = data["data"];
					};
					"done".postln;
					c.test=true;
					c.signal;
				};
				c.wait;
			};

			nslices.do {arg sliceidx;
				var k = "%-%".format(key, sliceidx);
				var slice_data = ();
				dumps.keys.do {|dkey|
					slice_data[dkey] = dumps[dkey][k];
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
			var data = size.collect {|i|["%-%".format(key, i), i]}.flatten;
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

	build { arg files, preSlices=nil, num_threads=1, action;
		var scaler_tree;
		forkIfNeeded {
			if (files.isString) {
				buffer = this.loadFolder(files);
			} {
				buffer = this.loadMultiple(files);
			};
			monos = this.loadBothMono;
			mono = this.toMono;
			if (preSlices.isNil) {
				slices = this.slice;
			} {
				slices = preSlices;
			};
			this.analyze(num_threads).keysValuesDo {|k, ds|
				s.sync;
				datasets[k] = ds;
				ds.cols({|count|
					datasets_info[k] = (cols: count);
				})
			};
			datasets.merged = this.mergeDatasets;
			s.sync;

			datasets.merged.cols({|count| datasets_info[\merged] = (cols: count) });

			//scaler_tree = this.fitKDTree;

			//scaler = scaler_tree[0];
			//kdtree = scaler_tree[1];
			//scaled_dataset = scaler_tree[2];
			//lookup = this.makeLookup;
			if (action.notNil) {
				action.(this);
			};
		}
	}

	store { arg path;
		var model = this;
		var name = key;
		path.mkdir;
		model.buffer.write(path +/+ "%.aiff".format(name));
		model.mono.write(path +/+ "%_mono.aiff".format(name));
		model.slices.write(path +/+ "%_slices.aiff".format(name), "aiff", "float");
		model.datasets.keys.do {|k|
			model.datasets[k].write(path +/+ "%_%_dataset.json".format(k, name));
		};
		model.labelset.write(path +/+ "%_labelset.json".format(name));
		FluidLoadStore.store(loader, path, name);
		//model.scaler.write(path +/+  "%_scaler.json".format(name));
		//model.kdtree.write(path +/+  "%_kdtree.json".format(name));
		//model.scaled_dataset.write(path +/+  "%_scaled_dataset.json".format(name));
		//model.lookup.write(path +/+  "%_lookup.json".format(name));
		File.use(path +/+ "%_slice_cache.scd".format(name), "w", {|f| f.write( model.slice_cache.asCompileString ) });
		File.use(path +/+ "%_datasets_info.scd".format(name), "w", {|f| f.write( model.datasets_info.asCompileString ) });
		"Stored % in %".format(name, path).postln;
	}

	load { arg path, loadDatasets=true;
		var name = key;
		buffer = Buffer.read(s, path +/+ "%.aiff".format(name));
		monos = [nil,nil];
		monos[0] = Buffer.readChannel(s, path +/+ "%.aiff".format(name), channels:[0]);
		monos[1] = Buffer.readChannel(s, path +/+ "%.aiff".format(name), channels:[1]);
		mono = Buffer.read(s, path +/+ "%_mono.aiff".format(name));
		slices = Buffer.read(s, path +/+ "%_slices.aiff".format(name));
		if (loadDatasets) {
			datasets.keys.do {|k|
				"loading %...".format(k).postln;
				datasets[k] = FluidDataSet(s).read(path +/+ "%_%_dataset.json".format(k, name), {"   % done.".format(k).postln});
			};
			//scaler = FluidNormalize(s, -1, 1).read(path +/+  "%_scaler.json".format(name));
			//kdtree = FluidKDTree(s).read(path +/+  "%_kdtree.json".format(name));
			//scaled_dataset = FluidDataSet(s).read(path +/+  "%_scaled_dataset.json".format(name));
			//lookup = FluidDataSet(s).read(path +/+  "%_lookup.json".format(name));
			labelset = FluidLabelSet(s).read(path +/+ "%_labelset.json".format(name));
		};
		loader = FluidLoadStore.load(path, name);
		slice_cache = (path +/+ "%_slice_cache.scd".format(name)).load;
		datasets_info = (path +/+ "%_datasets_info.scd".format(name)).load;

	}
}


FluidModelComposer {
	var <models, <model;

	*new { arg models=#[];
		^super.newCopyArgs(models).init;
	}

	prMergeModels {
		var key = "merged-%".format(this.models.collect {|m| m.key.asString}.join("_"));
		var datasets = ();

		model = FluidModel(key);

		forkIfNeeded {
			this.models.do {|m|
				m.datasets.keysValuesDo {|k,ds|
					if (datasets[k].isNil) {
						var query = FluidDataSetQuery(m.server);
						datasets[k] = FluidDataSet(m.server);
						query.addRange(0, m.datasets_info.postln[k].cols);
						query.transform(ds, datasets[k]);
					} {
						datasets[k].merge(ds);
					};
				};

				m.labelset.dump({|dic| dic["data"].keysValuesDo{|k,v| model.labelset.setLabel(k, v[0])}});

				model.slice_cache.putAll(m.slice_cache);
				m.server.sync;
			};
			datasets.keysValuesDo {|k, ds|
				model.datasets[k] = ds;
			};
		};
	}

	init {
		this.prMergeModels;
	}

	plot { arg dataset, algo=\pca, grid=1, action;
		^this.model.plot(dataset, algo, grid, action);
	}

	reduceDimension { arg n, algo=\pca, dataset=nil, grid=0, action;
		^this.model.reduceDimension(n, algo=\pca, dataset, grid=0, action);
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

FluiditySimplePlayer {
	var <>atk=0, <>rls=0, <>catk=0, <>crls=0, <>amp=1;

	*new {
		^super.newCopyArgs();
	}

	get {
		^{ arg corpus, slice;
			var info = corpus.slice_cache[slice.asSymbol];
			{
				Splay.ar(
					PlayBuf.ar(corpus.buffer.numChannels, corpus.buffer, 1, 1, info[1]) *
					Env.new([0,1,1,0], [atk,1-(atk + rls),rls], [catk, 1, crls]).ar(2, 1, info[0]/corpus.mono.sampleRate),
					1
				) * amp
			}.play
		}
	}

	rand {
		atk = 1.0.rand;
		rls = (1-atk).rand;
		catk = 10.0.rand2;
		crls = 10.rand2;
	}
}

FluidLoadStore {
	var <files, <index, <buffer;

	*new {|files, index, buffer|
		var soundfiles = files.collect {|path| SoundFile(path).info };
		^this.newCopyArgs(soundfiles, index, buffer);
	}

	*store { |loader, path, name|

		File.use(path +/+ "loader_files_%.scd".format(name), "w", {|file|
			file.write(loader.files.collect { |soundfile|
				soundfile.path
			}.asCompileString)
		});

		File.use(path +/+ "loader_index_%.scd".format(name), "w", {|file| file.write(loader.index.asCompileString)});
		loader.buffer.write(path +/+ "loader_buffer_%.aiff".format(name));
	}

	*load { |path, name|
		^this.new(
			(path +/+ "loader_files_%.scd".format(name)).load,
			(path +/+ "loader_index_%.scd".format(name)).load,
			Buffer.read(Server.default, path +/+ "%.aiff".format(name))
		);
	}
}

FluidLoadMultiple {
	var < files, idFunc,channelFunc;
	var < index;
	var < buffer;

	*new{ |files, idFunc, channelFunc |
		^super.newCopyArgs(files, idFunc,channelFunc);
	}

	play { |server, action|
		var sizes,channels,maxChan, startEnd,counter;
		server ?? {server = Server.default};
		sizes = files.collect{|f|f.numFrames()};
		channels = files.collect{|f| f.numChannels()};
		startEnd = sizes.inject([0],{|a,b| a ++ (b + a[a.size - 1])}).slide(2).clump(2);
		maxChan = channels[channels.maxIndex];
		counter = 0;
		index = IdentityDictionary();
		forkIfNeeded{
			buffer = Buffer.alloc(server,sizes.reduce('+'),maxChan);

			server.sync;
			buffer.updateInfo;
			buffer.query;
			server.sync;
			this.files.do{|f,i|
				var channelMap,identifier,entry;
				f.copyData(buffer, startEnd[i][0]);
				server.sync;
				buffer.updateInfo;
				server.sync;
				identifier = (f.path.basename).asSymbol;
				entry = IdentityDictionary();
				entry.add(\bounds->startEnd[i]);
				entry.add(\numchans->f.numChannels);
				entry.add(\sr->f.sampleRate);
				entry.add(\path->f.path);
				index.add(identifier->entry);
				counter = counter + 1;
				if(counter == (files.size)) {action !? action.value(index)};
			}
		};
	}
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

