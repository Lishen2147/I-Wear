"use client";

import React, { useState } from "react";
import UploadForm from "./subcomponents/Upload/UploadForm";
import Response from "./subcomponents/Upload/Response";
import Hero from "./Hero";

interface UploadProps {
  base64Image: string | null;
  setBase64Image: React.Dispatch<React.SetStateAction<string | null>>;
  isRequesting: boolean;
  setIsRequesting: React.Dispatch<React.SetStateAction<boolean>>;
  isLoading: boolean;
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>;
  faceStructure: string | null;
  setFaceStructure: React.Dispatch<React.SetStateAction<string | null>>;
  probabilities: number[] | null;
  setProbabilities: React.Dispatch<React.SetStateAction<number[] | null>>;
  generatedImages: string[] | null;
  setGeneratedImages: React.Dispatch<React.SetStateAction<string[] | null>>;
}

const Upload: React.FC<UploadProps> = ({
  base64Image,
  setBase64Image,
  isRequesting,
  setIsRequesting,
  isLoading,
  setIsLoading,
  faceStructure,
  setFaceStructure,
  probabilities,
  setProbabilities,
  generatedImages,
  setGeneratedImages,
}) => {
  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result;
        if (typeof result === "string") {
          setBase64Image(result);
        }
      };
      reader.readAsDataURL(file);
    }
  };
  return (
    <div className="flex flex-col items-center w-full">
      <div
        className="flex flex-row gap-8 items-center justify-center w-full p-12"
        style={{
          backgroundImage:
            "url(https://img.freepik.com/premium-photo/side-view-glasses-with-scenic-mountain-background_124507-261068.jpg)",
          backgroundSize: "cover",
          backgroundPosition: "center",
        }}
      >
        <Hero />
        <UploadForm
          onFileChange={handleImageChange}
          image={base64Image}
          setFaceStructure={setFaceStructure}
          setIsLoading={setIsLoading}
          setIsRequesting={setIsRequesting}
          setProbabilities={setProbabilities}
          setGeneratedImages={setGeneratedImages}
        />
      </div>
      <div className="flex flex-col items-center gap-2 w-full p-12 bg-gradient-to-r from-blue-300 via-purple-300 to-slate-200">
        <div className="flex flex-col items-center gap-2">
          <h1 className="text-3xl font-semibold text-black">I-Wear Viewer</h1>
          <p className="text-lg text-black">
            Preview our best selections curated to complement your features
          </p>
        </div>
        <Response
          isLoading={isLoading}
          isRequesting={isRequesting}
          faceStructure={faceStructure}
          generatedImages={generatedImages}
          probabilities={probabilities}
        />
      </div>
    </div>
  );
};

export default Upload;
