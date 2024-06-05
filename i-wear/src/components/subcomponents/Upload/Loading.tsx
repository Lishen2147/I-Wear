"use client";

import React from "react";
import PredictionImages from "@/components/PredictionImages";

interface LoadingProps {
  isLoading: boolean;
  faceStructure: string | null;
  generatedImages: string[] | null;
  probabilities: number[] | null;
}

const Loading: React.FC<LoadingProps> = ({
  isLoading,
  faceStructure,
  generatedImages,
  probabilities,
}) => {
  return (
    <div>
      {isLoading ? (
        <div className="text-xl font-semibold">Loading...</div>
      ) : (
        <div className="flex flex-col gap-4">
          <div className="text-2xl font-bold underline underline-offset-4 text-black">
            {"Facial Structure: " + faceStructure}
          </div>
          {generatedImages && (
            <PredictionImages
              images={generatedImages}
              probabilities={probabilities}
            />
          )}
        </div>
      )}
    </div>
  );
};

export default Loading;
