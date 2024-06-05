"use client";

import React from "react";
import Loading from "./Loading";

interface ResponseProps {
  isLoading: boolean;
  isRequesting: boolean;
  faceStructure: string | null;
  generatedImages: string[] | null;
  probabilities: number[] | null;
}

const Response: React.FC<ResponseProps> = ({
  isLoading,
  isRequesting,
  faceStructure,
  generatedImages,
  probabilities,
}) => {
  return (
    <div>
      {!isRequesting ? (
        <div className="text-md font-semibold text-black">
          Input a face to get started!
        </div>
      ) : (
        <Loading
          isLoading={isLoading}
          faceStructure={faceStructure}
          generatedImages={generatedImages}
          probabilities={probabilities}
        />
      )}
    </div>
  );
};

export default Response;
