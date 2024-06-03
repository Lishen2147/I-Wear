"use client";
import MODEL_CONSTANTS from "../../../constants.json";
import React, { useState } from "react";
import Header from "../components/Header";
import Upload from "../components/Upload";
import PredictionProbabilties from "@/components/PredictionProbabilities";


export default function Home() {
  const MODEL_CLASSSES = MODEL_CONSTANTS.model_params.facial_structure_classes
  const [base64Image, setBase64Image] = useState<string | null>(null);
  const [isRequesting, setIsRequesting] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [faceStructure, setFaceStructure] = useState<string | null>(null);
  const [probabilities, setProbabilities] = useState<number[] | null>(null);
  const [generatedImages, setGeneratedImages] = useState<string[] | null> (null);
  return (
    <main className="bg-black w-screen h-screen">
      <Header />
      <Upload base64Image={base64Image} isRequesting={isRequesting} isLoading={isLoading} faceStructure={faceStructure} setBase64Image={setBase64Image} setIsRequesting={setIsRequesting} setIsLoading={setIsLoading} setFaceStructure={setFaceStructure} setProbabilities={setProbabilities} generatedImages={generatedImages} setGeneratedImages={setGeneratedImages} />
      <PredictionProbabilties probabilities={probabilities} classes={MODEL_CLASSSES} />
    </main>
  );
}
