"use client";

import React from "react";
import Image from "next/image";

interface UploadFormProps {
  image: string | null;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>;
  setFaceStructure: React.Dispatch<React.SetStateAction<string | null>>;
  setIsRequesting: React.Dispatch<React.SetStateAction<boolean>>;
  setProbabilities: React.Dispatch<React.SetStateAction<number[] | null>>;
  setGeneratedImages: React.Dispatch<React.SetStateAction<string[] | null>>;
}

const UploadForm: React.FC<UploadFormProps> = ({
  image,
  onFileChange,
  setIsLoading,
  setFaceStructure,
  setIsRequesting,
  setProbabilities,
  setGeneratedImages,
}) => {
  const POST_URL = "http://localhost:8080/predict";
  const GENERATE_IMAGE_URL = "http://localhost:8080/glasses";

  const formSubmitHandler = async (e: React.SyntheticEvent) => {
    e.preventDefault();
    try {
      setIsRequesting(true);
      setIsLoading(true);
      const data = await fetch(POST_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image }),
      });
      const response = await data.json();

      setIsLoading(false);
      if (response.response_code != 200) {
        setIsRequesting(false);
        setFaceStructure(null);
        return;
      }
      // setTimeout(() => {
      //   setIsRequesting(false);
      //   setFaceStructure(null);
      //   setProbabilities(response.prediction_probabilities);
      // }, 20000);
      setIsLoading(true);
      const gen_data = await fetch(GENERATE_IMAGE_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image, prediction: response.prediction }),
      });
      const img_response = await gen_data.json();
      setIsLoading(false);
      if (img_response.images.length === 0) {
        setIsRequesting(false);
        setFaceStructure(null);
        setProbabilities(null);
        setGeneratedImages(null);
        return;
      }
      setFaceStructure(response.prediction);
      setProbabilities(response.prediction_probabilities);
      setGeneratedImages(img_response.images);
    } catch {
      setIsRequesting(false);
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center gap-2 bg-slate-200 p-4 rounded-lg border-2 border-yellow-500">
      <div className="flex flex-col gap-4">
        <h2 className="font-bold text-2xl underline underline-offset-1 text-black">
          Try it out!
        </h2>
        {image ? (
          <Image
            className="rounded-md border-indigo-500 border-4 object-cover object-center"
            src={image}
            width={450}
            height={450}
            alt="Uploaded"
          />
        ) : (
          <div className="w-60 h-60 bg-gray-300 rounded-md border-indigo-500 border-4 flex items-center justify-center">
            <div className="w-[95%] h-[95%] bg-black flex items-center justify-center">
              <p className="text-white text-center text-4xl animate-pulse">
                D:
              </p>
            </div>
          </div>
        )}
      </div>
      <div className="flex flex-col gap-4 max-w-60">
        <input
          className="text-md"
          type="file"
          accept="image/*"
          onChange={onFileChange}
        />
        <button
          className="bg-indigo-400 text-center text-lg p-1 text-white rounded-lg hover:bg-indigo-500"
          type="submit"
          onClick={formSubmitHandler}
        >
          GET SUGGESTIONS
        </button>
      </div>
    </div>
  );
};

export default UploadForm;
