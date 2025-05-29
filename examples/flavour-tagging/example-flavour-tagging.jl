# Import required packages
using EDM4hep
using EDM4hep.RootIO
using JetReconstruction
using LorentzVectorHEP
using JSON
using ONNXRunTime
using StructArrays

"""
Main function to perform jet flavor tagging
"""
function main()
    # Paths to model files
    model_dir = "data/wc_pt_7classes_12_04_2023"
    onnx_path = joinpath(model_dir, "fccee_flavtagging_edm4hep_wc_v1.onnx")
    json_path = joinpath(model_dir, "fccee_flavtagging_edm4hep_wc_v1.json")

    # Load the configuration and model
    config = JSON.parsefile(json_path)
    model = ONNXRunTime.load_inference(onnx_path)

    # Display the output classes we'll predict
    println("The model predicts these flavor classes:")
    for class_name in config["output_names"]
        println(" - ", class_name)
    end

    # Path to ROOT file with EDM4hep data
    edm4hep_path = "data/events_080263084.root"
    reader = RootIO.Reader(edm4hep_path)

    # Get event information
    events = RootIO.get(reader, "events")
    println("Loaded $(length(events)) events")

    # Choose a specific event to analyze (event #13)
    event_id = 13
    evt = events[event_id]
    println("Processing event #$event_id")

    # Get reconstructed particles and tracks
    recps = RootIO.get(reader, evt, "ReconstructedParticles")
    tracks = RootIO.get(reader, evt, "EFlowTrack_1")

    # Get needed collections for feature extraction
    bz = RootIO.get(reader, evt, "magFieldBz", register = false)
    trackdata = RootIO.get(reader, evt, "EFlowTrack")
    trackerhits = RootIO.get(reader, evt, "TrackerHits")
    gammadata = RootIO.get(reader, evt, "EFlowPhoton")
    nhdata = RootIO.get(reader, evt, "EFlowNeutralHadron")
    calohits = RootIO.get(reader, evt, "CalorimeterHits")
    dNdx = RootIO.get(reader, evt, "EFlowTrack_2")
    track_L = RootIO.get(reader, evt, "EFlowTrack_L", register = false)

    println("Loaded $(length(recps)) reconstructed particles")
    println("Loaded $(length(tracks)) tracks")

    # Cluster jets using the EEkt algorithm with R=2.0 and p=1.0
    cs = jet_reconstruct(recps; p = 1.0, R = 2.0, algorithm = JetAlgorithm.EEKt)

    # Get 2 exclusive jets
    jets = exclusive_jets(cs; njets=2, T=EEJet)

    # For each jet, get its constituent particles
    constituent_indices = [constituent_indexes(jet, cs) for jet in jets]
    jet_constituents = build_constituents_cluster(recps, constituent_indices)

    println("Extracting features for flavor tagging...")
    feature_data = extract_features(
        jets, 
        jet_constituents, 
        tracks, 
        bz, 
        track_L, 
        trackdata, 
        trackerhits, 
        gammadata, 
        nhdata, 
        calohits, 
        dNdx
    )
    println("Step 1: Feature extraction completed.")

    model, config = setup_weaver(
        onnx_path,
        json_path
    )

    println("Step 2: Weaver setup completed.")

    input_tensors = prepare_input_tensor(
        jet_constituents,
        jets,
        config,
        feature_data
    )

    println("Step 3: Input tensor preparation completed.")

    println("Running flavor tagging inference...")
    weights = get_weights(
        0,  # Thread slot
        feature_data,
        jets,
        jet_constituents,
        config,
        model
    )
    println("Step 4: Weights retrieval completed.")

    jet_scores = Dict{String, Vector{Float32}}()
    for (i, score_name) in enumerate(config["output_names"])
        jet_scores[score_name] = get_weight(weights, i-1)
    end

    println("Jet scores:")
    for (name, scores) in jet_scores
        println(" - $name: $(scores[1])")
    end
end

# Execute the main function if this file is run as a script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end