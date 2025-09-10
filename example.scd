(
s.waitForBoot {
	// set to your own folder
	//   var folder = "/some/path";
	// this will load all files that come with FluCoMa
	var folder = FluidFilesPath();

	// increase the number of threads to make the slicing faster,
	f = FluidModel();
	f.build(folder, num_threads: 10, action: {
		// after done, plot the merged dataset, colors represent the original source file
		f.plot
	});
}
)

// try other plots:
f.plot(\mfcc);

// by crhoma
f.plot(\chroma);


// plot and play
(
f.plot(\mfcc, {arg slice, dur_buf;
	{Splay.ar(PlayBuf.ar(2, f.buffer, 1, 1, Index.kr(dur_buf, 1), 0, 0) * Env.new([0,1,1,0],[0,Index.kr(dur_buf, 0),0]).ar(2))}.play
})
)