#include <cstdlib>
#include <iostream>
#include "glog/logging.h"
#include <chrono>
#include <vector>
#include <fstream>
#include <algorithm>
#include <tuple>
#include <queue>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/impl/vtk_lib_io.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
#include "ceres/ceres.h"
#include "class/data_path.h"
#include "class/data_structure.h"
#include "class/visualize.h"
#include "class/reconstruction.h"
#include "class/feature_matching.h"
#include "class/ranking_system.h"
#include "class/genetic_algorithm.h"

//------------------------------------------------------------------------------------------------------------------------//

//#define NO_RIM_INFO
#define NO_BASE_INFO

using namespace std;
using namespace Eigen;

vector<Geom> shard(SHARD_NUMBER);
vector<Trans> GT_trans(SHARD_NUMBER);
MatrixXd GT_graph(SHARD_NUMBER, SHARD_NUMBER);

pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Pot reconstruction"));

VisSwitchVariables vis;

//------------------------------------------------------------------------------------------------------------------------//

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* nothing)
{
	pcl::visualization::PCLVisualizer* viewer = static_cast<pcl::visualization::PCLVisualizer*> (nothing);
	std::string key_string = event.getKeySym();
	bool key_down = event.keyDown();
	vis.KeyEvent(key_string, key_down);
}

//------------------------------------------------------------------------------------------------------------------------//

int main(int argc, char** argv)
{
	//-------------------------------------------------------------------------------------------------------------------//
	//#################### PCL viewer setting ####################//
	double calculation_time(0);
	double time_ga(0);

	viewer->setBackgroundColor(0, 0, 0);
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	viewer->registerKeyboardCallback(&keyboardEventOccurred, (void*)viewer.get());

	enum class PostGaIcpMode {
		GlobalFine,
		Compare
	};

	PostGaIcpMode post_icp_mode = PostGaIcpMode::GlobalFine;

	for (int arg_i = 1; arg_i < argc; ++arg_i) {
		string arg = argv[arg_i];
		if (arg == "--icp-global") {
			post_icp_mode = PostGaIcpMode::GlobalFine;
		}
		else if (arg == "--icp-compare") {
			post_icp_mode = PostGaIcpMode::Compare;
		}
		else if (arg.rfind("--icp-mode=", 0) == 0) {
			string mode = arg.substr(11);
			if (mode == "global") {
				post_icp_mode = PostGaIcpMode::GlobalFine;
			}
			else if (mode == "compare") {
				post_icp_mode = PostGaIcpMode::Compare;
			}
			else {
				cout << "[WARN] Unknown --icp-mode value: " << mode
					 << " (expected global|compare). Using default global." << endl;
			}
		}
	}

	auto IcpModeName = [](PostGaIcpMode mode) -> string {
		switch (mode) {
		case PostGaIcpMode::GlobalFine:
			return "global";
		case PostGaIcpMode::Compare:
			return "compare";
		default:
			return "unknown";
		}
	};

	cout << "[ICP MODE] " << IcpModeName(post_icp_mode)
		 << " (flags: --icp-mode=global|compare)" << endl;

	//------------------------------------------------------------------------------------------------------------------//

	cout << "#################### Pottery Data load ####################" << endl;

	int max_breakline_points(0);

	for (int i = 0; i < SHARD_NUMBER; i++) {
		shard[i].edge_line_.ReadAxis(axis_path[i]);

		if (shard[i].edge_line_.axis_point_.empty()) {
			shard_on_off[i] = false;
			continue;
		}

		// Consider multi-axis shards at the same time
		if(shard_on_off[i])	{
			shard[i].edge_line_.ReadPCDFileWithInfo(file_path[i]);

			if (shard[i].edge_line_.point_.cols() < 50) {
				shard_on_off[i] = false;
				shard[i].edge_line_.Remove();
				continue;
			}

			shard[i].edge_line_.CalculateLineNormal();
			int breakline_points = shard[i].edge_line_.point_.cols();
			max_breakline_points = max(max_breakline_points, breakline_points);
			shard[i].LoadSurface(surface_in[i], surface_out[i], surface_fr[i]);
			shard[i].is_matching_ = true;
			shard[i].sur_frac_.CalculateLineNormal();
		}
	}

	//------------------------------------------------------------------------------------------------------------------//

#ifdef NO_RIM_INFO
	for (int i = 0; i < SHARD_NUMBER; i++) {
		shard[i].edge_line_.is_seg_rim_ = false;
	}
#endif

#ifdef NO_BASE_INFO
	for (int i = 0; i < SHARD_NUMBER; i++) {
		shard[i].edge_line_.is_seg_base_ = false;
	}
#endif

	//------------------------------------------------------------------------------------------------------------------//


	cout << "#################### Ground Truth data load ####################" << endl;

	for (int i = 0; i < SHARD_NUMBER; i++) {
		for (int j = 0; j < SHARD_NUMBER; j++)
			GT_graph(i, j) = 0;
	}

	int start_index(0);

	for (int i = 0; i < NUM_MIXED_SHERD; i++) {
		MatrixXd single_graph;
		ifstream myfile(gt_graph_path[i]);
		string str;
		vector<string> fileContents;
		stringstream ss;

		while (getline(myfile, str)) {
			fileContents.push_back(str);
		}

		int num_raw = fileContents.size();
		single_graph.resize(num_raw, num_raw);

		for (int j = 0; j < num_raw; j++) {
			ss << fileContents[j];

			for (int k = 0; k < num_raw; k++) {
				ss >> single_graph(j, k);
			}
			ss.clear();
		}

		for (int j = start_index; j < num_raw + start_index; j++) {
			GT_trans[j].Read(gt_T_path[j]);

			for (int k = start_index; k < num_raw + start_index; k++) {
				GT_graph(j, k) = single_graph(j - start_index, k - start_index);
			}
		}

		start_index += num_raw;
	}

	//########## Remove excluded sherd information
	for (int i = 0; i < SHARD_NUMBER; i++) {
		if (!shard_on_off[i]) {
			for (int j = 0; j < SHARD_NUMBER; j++) {
				GT_graph(i, j) = 0;
				GT_graph(j, i) = 0;
			}
		}
	}
	cout << GT_graph << endl;

	//------------------------------------------------------------------------------------------------------------------//

	cout << "#################### Save initial state ####################" << endl;

	vector<Visualize> pc_origin(SHARD_NUMBER);

	for (int i = 0; i < SHARD_NUMBER; i++) {
		if (shard_on_off[i]) {
			std::string pointName = "origin_" + std::to_string(i + 1);
			pc_origin[i].MakePointCloud(shard[i].edge_line_.point_, shard[i].edge_line_.normal_, pointName);
			pointName = "o_Mesh" + std::to_string(i + 1);
			pc_origin[i].MakeMesh(obj_path[i], pointName);
		}
	}

	//------------------------------------------------------------------------------------------------------------------//

	auto start_time_total = std::chrono::high_resolution_clock::now();

	//------------------------------------------------------------------------------------------------------------------//

	// Keep a copy of original RAW shard positions
	vector<Geom> shard_raw_backup = shard;

	cout << "#################### Change Axis symmetrix to z axis ####################" << endl;

	vector<Trans> T_axis(SHARD_NUMBER);

	for (int i = 0; i < SHARD_NUMBER; i++) {

		if (shard[i].is_matching_) {
			Matrix3d R_d = Matrix3d::Identity();
			Vector3d t_d = { 0, 0, 0 };
			T_axis[i].Set(R_d, t_d, i + 1, i + 1);

			// Align symmetric axis to z-axis
			AxisAlignment(shard[i].edge_line_, R_d, t_d);
			shard[i].SurMove(R_d, t_d, true);

			pc_origin[i].UpdateData(viewer, shard[i].edge_line_.point_, shard[i].edge_line_.normal_);
			pc_origin[i].AddPointCloud(viewer);
			pc_origin[i].MeshTransform(R_d, t_d, viewer);
			pc_origin[i].AddMesh(viewer);
			T_axis[i].Input(R_d, t_d);		// Save transformation matrix to z-axis

			CalculateFeatureAxisless(shard[i]);

			//######## Multi axis
			int num_axis = shard[i].edge_line_.axis_norm_.size();

			if (num_axis > 1) {

				for (int j = 1; j < num_axis; j++) {
					Matrix3d R_a, R_i;
					Vector3d t_a, t_i;
					AxisAlignment(shard[i].edge_line_, R_a, t_a, j);
					CalculateFeatureAxisless(shard[i], j);
					R_i = R_a.inverse();
					t_i = -R_i * t_a;
					shard[i].MoveWOSurface(R_i, t_i);
				}
			}
		}
	}

	//------------------------------------------------------------------------------------------------------------------//

	cout << "#################### Feature matching ####################" << endl;

	list<LCSIndex> LCS_out;
	FeatureComp(shard, LCS_out, 25, MINIMUM_NUMBER, 0);
	cout << "Total Number : " << LCS_out.size() << endl;

	//------------------------------------------------------------------------------------------------------------------//

	cout << "#################### Pairwise pruning ####################" << endl;

	PairwisePruning(shard, LCS_out);

	cout << "Pruned match count: " << LCS_out.size() << endl;


	// Save matches computed on original axis-aligned positions
	// (LCS_out gets overwritten during iterative GA, so preserve it here)

	list<LCSIndex> LCS_original = LCS_out;

	//------------------------------------------------------------------------------------------------------------------//

	cout << "#################### Genetic Algorithm search ####################" << endl;

	// Iterative GA Parameters
	int active_shard_count = 0;
	for (int i = 0; i < SHARD_NUMBER; ++i) {
		if (shard_on_off[i]) active_shard_count++;
	}
	const int kMaxGAIterations = 1;

	vector<Trans> T_ga;
	MatrixXd graph_ga = MatrixXd::Zero(SHARD_NUMBER, SHARD_NUMBER);
	vector<Trans> T_ga_eval = T_axis;
	vector<Trans> T_ga_vis(SHARD_NUMBER); // Tracks overall best GA movement (axis->assembled)
	vector<Trans> T_live(SHARD_NUMBER);   // The currently exploring accumulated transform (axis->assembled)
	vector<Trans> T_best(SHARD_NUMBER);   // The best accumulated transform
	vector<Matrix3d> R_preview_applied(SHARD_NUMBER, Matrix3d::Identity());
	vector<Vector3d> t_preview_applied(SHARD_NUMBER, Vector3d::Zero());
	double best_fitness_so_far = -1e9;
	int ga_iteration = 0;
	int patience_counter = 0;

	// Keep a copy of original axis-aligned shard positions
	// so we can reset between iterations cleanly
	vector<Geom> shard_original = shard;
	vector<Vector3d> sherd_label_anchor_local(SHARD_NUMBER, Vector3d::Zero());
	vector<bool> sherd_label_valid(SHARD_NUMBER, false);

	for (int i = 0; i < SHARD_NUMBER; ++i) {
		if (!shard_on_off[i]) {
			continue;
		}

		const MatrixXd& pts = shard_original[i].edge_line_.point_;
		if (pts.cols() <= 0) {
			continue;
		}

		Vector3d centroid = Vector3d::Zero();
		for (int c = 0; c < pts.cols(); ++c) {
			centroid += pts.col(c);
		}
		centroid /= static_cast<double>(pts.cols());
		sherd_label_anchor_local[i] = centroid;
		sherd_label_valid[i] = true;
	}

	auto ClearSherdLabels = [&]() {
		for (int i = 0; i < SHARD_NUMBER; ++i) {
			if (!shard_on_off[i]) {
				continue;
			}
			viewer->removeShape("sherd_label_" + to_string(i + 1));
		}
	};

	auto ShowSherdLabels = [&](const vector<Trans>& transforms, double r, double g, double b) {
		ClearSherdLabels();
		const double kLabelScale = 5.0;
		for (int i = 0; i < SHARD_NUMBER; ++i) {
			if (!shard_on_off[i] || !sherd_label_valid[i]) {
				continue;
			}

			Matrix3d R = Matrix3d::Identity();
			Vector3d t = Vector3d::Zero();
			transforms[i].Output(R, t);
			Vector3d p = R * sherd_label_anchor_local[i] + t;
			pcl::PointXYZ label_pt(static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2]));
			viewer->addText3D(to_string(i + 1),
				label_pt,
				kLabelScale,
				r,
				g,
				b,
				"sherd_label_" + to_string(i + 1));
		}
	};

	auto start_time_ga = std::chrono::high_resolution_clock::now();

	for (ga_iteration = 0; ga_iteration < kMaxGAIterations; ++ga_iteration) {

		cout << "=== GA Iteration " << ga_iteration + 1 << " / " << kMaxGAIterations << " ===" << endl;

		// Run GA on current match list, selecting the full assembly size
		int target_edges = active_shard_count - 1;
		GeneticAssembler ga_iter(shard, LCS_out, SHARD_NUMBER, target_edges);
		// ga_iter.RunBeamSearchGT(GT_graph, GT_trans, T_axis);
		ga_iter.Run(GT_graph, GT_trans, T_axis);
		T_ga = ga_iter.GetTransforms();

		// Update live transformation directly to the absolute pose T_ga
		for (int i = 0; i < SHARD_NUMBER; ++i) {
			if (!shard_on_off[i]) continue;

			Matrix3d R_delta;
			Vector3d t_delta;
			T_ga[i].Output(R_delta, t_delta);
			T_live[i].Set(R_delta, t_delta);
		}

		// Get best fitness from this run
		double current_fitness = ga_iter.GetBestFitness();
		double improvement = current_fitness - best_fitness_so_far;
		bool improved = (current_fitness > best_fitness_so_far);

		// Only keep state if it strictly improves the global best fitness.
		if (improved) {
			best_fitness_so_far = current_fitness;

			graph_ga = ga_iter.GetGraph();
			T_best = T_live; // Snapshot the best axis-aligned transform

			for (int i = 0; i < SHARD_NUMBER; ++i) {
				if (!shard_on_off[i]) {
					continue;
				}

				Matrix3d R_b;
				Vector3d t_b;
				T_best[i].Output(R_b, t_b);

				// T_ga_eval tracks Raw -> Axis -> Assembled
				T_ga_eval[i] = T_axis[i];
				T_ga_eval[i].Input(R_b, t_b);
				T_ga_vis[i] = T_best[i];
			}
		}

		cout << "[GA Iter " << ga_iteration + 1 << "] "
			 << "Best fitness: " << current_fitness
			 << " (improvement: " << improvement << ")" << endl;
	}

	auto end_time_ga = std::chrono::high_resolution_clock::now();
	time_ga = std::chrono::duration<double>(end_time_ga - start_time_ga).count();

	cout << "GA converged after " << ga_iteration + 1 << " iteration(s)." << endl;

	pair<int, int> sherd_acc, edge_acc;
	vector<bool> right_sherd_ga(SHARD_NUMBER, true);

	for (int i = 0; i < SHARD_NUMBER; ++i) {
		if (!shard_on_off[i]) {
			right_sherd_ga[i] = false;
		}
	}

	auto [k_sherd, t_sherd, k_edge, t_edge] = CountResult(
		GT_graph, GT_trans, graph_ga, T_ga_eval, right_sherd_ga);

	auto PrintStage = [](const string& label,
		int k_sherd_stage,
		int t_sherd_stage,
		int k_edge_stage,
		int t_edge_stage)
		{
			cout << "########## " << label << " ##########" << endl;
			cout << "Sherd Accuracy : " << k_sherd_stage << " / " << t_sherd_stage << endl;
			cout << "Edge Accuracy  : " << k_edge_stage << " / " << t_edge_stage << endl;
			cout << "########################################" << endl;
		};

	PrintStage("Pre-ICP GA Results", k_sherd, t_sherd, k_edge, t_edge);

	int k_sherd_global = k_sherd, t_sherd_global = t_sherd;
	int k_edge_global = k_edge, t_edge_global = t_edge;
	vector<Trans> T_ga_eval_global = T_ga_eval;
	vector<Trans> T_ga_vis_global = T_ga_vis;
	MatrixXd graph_ga_global = graph_ga;
	bool has_global_result = false;


	//------------------------------------------------------------------------------------------------------------------//
	if (post_icp_mode == PostGaIcpMode::GlobalFine || post_icp_mode == PostGaIcpMode::Compare) {
		cout << "#################### Post-GA Global ICP Refinement ####################" << endl;

		vector<Geom> shard_fine = shard_original;
		vector<Matrix3d> R_fine(SHARD_NUMBER, Matrix3d::Identity());
		vector<Vector3d> t_fine(SHARD_NUMBER, Vector3d::Zero());
		vector<bool> true_node_ga(SHARD_NUMBER, false);

		for (int i = 0; i < SHARD_NUMBER; i++) {
			if (!shard_on_off[i]) continue;
			true_node_ga[i] = true;
			Matrix3d R = Matrix3d::Identity();
			Vector3d t = Vector3d::Zero();
			T_ga_vis[i].Output(R, t);
			shard_fine[i].Move(R, t, true);
		}

		graph_ga_global = graph_ga;
		IcpFine(shard_fine, R_fine, t_fine, true_node_ga, graph_ga_global);

		T_ga_eval_global = T_ga_eval;
		T_ga_vis_global = T_ga_vis;
		for (int i = 0; i < SHARD_NUMBER; i++) {
			if (!true_node_ga[i]) continue;
			T_ga_eval_global[i].Input(R_fine[i], t_fine[i]);
			T_ga_vis_global[i].Input(R_fine[i], t_fine[i]);
		}

		vector<bool> right_sherd_fine(SHARD_NUMBER, true);
		for (int i = 0; i < SHARD_NUMBER; i++) {
			if (!shard_on_off[i]) right_sherd_fine[i] = false;
		}

		tie(k_sherd_global, t_sherd_global, k_edge_global, t_edge_global) = CountResult(
			GT_graph, GT_trans, graph_ga_global, T_ga_eval_global, right_sherd_fine);

		has_global_result = true;
		PrintStage("Post-ICP Global Results", k_sherd_global, t_sherd_global, k_edge_global, t_edge_global);
	}

	vector<Trans> T_final_vis = T_ga_vis;
	string final_stage_label = "Pre-ICP GA";
	int k_sherd_final = k_sherd, t_sherd_final = t_sherd;
	int k_edge_final = k_edge, t_edge_final = t_edge;

	auto IsResultBetter = [](int sherd_a, int edge_a, int sherd_b, int edge_b) {
		if (sherd_a != sherd_b) {
			return sherd_a > sherd_b;
		}
		return edge_a > edge_b;
	};

	if (post_icp_mode == PostGaIcpMode::GlobalFine && has_global_result) {
		final_stage_label = "Post-ICP Global";
		k_sherd_final = k_sherd_global;
		t_sherd_final = t_sherd_global;
		k_edge_final = k_edge_global;
		t_edge_final = t_edge_global;
		T_final_vis = T_ga_vis_global;
	}
	else if (post_icp_mode == PostGaIcpMode::Compare) {
		bool selected_non_pre = false;

		if (has_global_result &&
			IsResultBetter(k_sherd_global, k_edge_global, k_sherd_final, k_edge_final)) {
			final_stage_label = "Post-ICP Global (selected in compare)";
			k_sherd_final = k_sherd_global;
			t_sherd_final = t_sherd_global;
			k_edge_final = k_edge_global;
			t_edge_final = t_edge_global;
			T_final_vis = T_ga_vis_global;
			selected_non_pre = true;
		}

		if (!selected_non_pre) {
			final_stage_label = "Pre-ICP GA (selected in compare)";
		}
	}



	// Final result summary for refined assembly
	sherd_acc = { k_sherd_final, t_sherd_final };
	edge_acc = { k_edge_final, t_edge_final };
	auto end_time_total = std::chrono::high_resolution_clock::now();
	double time_total = std::chrono::duration<double>(end_time_total - start_time_total).count();

	cout << "########## Final Refined Results ##########" << endl;
	cout << "Selected Stage      : " << final_stage_label << endl;
	cout << "Final Sherd Accuracy : " << k_sherd_final << " / " << t_sherd_final << endl;
	cout << "Final Edge Accuracy  : " << k_edge_final << " / " << t_edge_final << endl;
	cout << "Total Runtime        : " << time_total << " sec" << endl;
	cout << "GA Runtime           : " << time_ga << " sec" << endl;
	cout << "###########################################" << endl;

	string path_result_refined = path + "Result/Refined_";
	string refined_mkdir_cmd = "mkdir -p \"" + path_result_refined + "\"";
	std::system(refined_mkdir_cmd.c_str());
	SaveAcc(path_result_refined, sherd_acc, edge_acc, time_total);

	for (int i = 0; i < SHARD_NUMBER; i++) {
		pc_origin[i].TurnOffData(viewer);
	}
	ClearSherdLabels();

	// Build Ground Truth assembly for comparison
	vector<Visualize> pc_gt(SHARD_NUMBER);
	for (int i = 0; i < SHARD_NUMBER; i++) {
		if (!shard_on_off[i]) continue;
		string pointName = "gt_origin_" + to_string(i + 1);
		pc_gt[i].MakePointCloud(shard_original[i].edge_line_.point_, shard_original[i].edge_line_.normal_, pointName);
		pointName = "gt_Mesh" + to_string(i + 1);
		pc_gt[i].MakeMesh(obj_path[i], pointName);

		Matrix3d R_gt;
		Vector3d t_gt;
		GT_trans[i].Output(R_gt, t_gt);
		pc_gt[i].MeshTransform(R_gt, t_gt, viewer);
		pc_gt[i].TurnOffData(viewer);
	}

	// Apply selected final transforms and show initial result.
	for (int i = 0; i < SHARD_NUMBER; i++) {
		if (!shard_on_off[i]) continue;
		Matrix3d R_vis = Matrix3d::Identity();
		Vector3d t_vis = Vector3d::Zero();
		T_final_vis[i].Output(R_vis, t_vis);
		pc_origin[i].UpdateData(viewer,
			shard_original[i].edge_line_.point_,
			shard_original[i].edge_line_.normal_);
		pc_origin[i].Transform(R_vis, t_vis, viewer);
		pc_origin[i].AddPointCloud(viewer);
		pc_origin[i].AddMesh(viewer);
	}
	ShowSherdLabels(T_final_vis, 1.0, 1.0, 0.2);

	viewer->resetCamera();

	// Interactive toggle loop — press 'G' to switch between GA and GT
	bool last_ground = false; // Monitor toggle state
	cout << "Showing Refined assembly. Press 'G' to toggle Ground Truth comparison! Press 'Q' to quit." << endl;

    while (!viewer->wasStopped()) {
		if (vis.ground_ != last_ground) {
			for (int i = 0; i < SHARD_NUMBER; i++) {
				if (!shard_on_off[i]) continue;
				if (vis.ground_) {
					// SHOW GROUND TRUTH (Green tint)
					pc_origin[i].TurnOffData(viewer);
					pc_gt[i].AddMesh(viewer, 0.0, 1.0, 0.5); // Green-ish Cyan
				} else {
					// SHOW GA ASSEMBLY (Default)
					pc_gt[i].TurnOffData(viewer);
					pc_origin[i].AddMesh(viewer, 0.8, 0.8, 0.8); // Standard Grey
				}
			}

			if (vis.ground_) {
				ShowSherdLabels(GT_trans, 0.0, 1.0, 0.5);
			}
			else {
				ShowSherdLabels(T_final_vis, 1.0, 1.0, 0.2);
			}

			last_ground = vis.ground_;
		}
		viewer->spinOnce(100);
	}

	//------------------------------------------------------------------------------------------------------------------//

	return 0;
}
