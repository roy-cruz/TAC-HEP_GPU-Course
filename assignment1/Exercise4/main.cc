#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "t1.h"

#include <TMath.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h> 
#include <TLorentzVector.h>

// Class prototypes

class Particle{
	public:
		Particle();
		// FIXME : Create an additional constructor that takes 4 arguments --> the 4-momentum
		Particle(Float_t px, Float_t py, Float_t pz, Float_t energy);

		// Added factory since data provided is in pt, eta, phi, E format
		static Particle FromPtEtaPhiE(Float_t pt, Float_t eta, Float_t phi, Float_t energy);
		
		// Member variables
		Float_t   pt, eta, phi, E, m, p[4]; 

		// p[0] -> E
		// p[1] -> px
		// p[2] -> py
		// p[3] -> pz

		// Methods
		void     p4(Float_t pt, Float_t eta, Float_t phi, Float_t energy);
		void     print();
		void     setMass(Float_t mass) { m = mass; };
		Float_t   sintheta();
};

class Lepton : public Particle {
	public: 
		Lepton() : Particle(), Q(0) {};
		Lepton(Float_t px, Float_t py, Float_t pz, Float_t energy, Int_t charge) : Particle(px, py, pz, energy), Q(charge) {};
		static Lepton FromPtEtaPhiE(Float_t pt, Float_t eta, Float_t phi, Float_t energy, Int_t charge);

		Int_t 	Q;	
		void	setCharge(Int_t charge) { Q = charge; };
};

class Jet : public Particle {
	public: 
		Jet() : Particle(), hadronFlav(-1) {}; // -1 here means undefined/unspecified
		Jet(Float_t px, Float_t py, Float_t pz, Float_t energy, Int_t hadronFlavor) : Particle(px, py, pz, energy), hadronFlav(hadronFlavor) {};
 		static Jet FromPtEtaPhiE(Float_t pt, Float_t eta, Float_t phi, Float_t energy, Int_t hadronFlavor);

		Int_t 	hadronFlav;
		void	setHadronFlavor(Int_t flavor) { hadronFlav = flavor; };
};

//*****************************************************************************
//                                                                             *
//    MEMBERS functions of the Particle Class                                  *
//                                                                             *
//*****************************************************************************

Particle::Particle(){
	pt = eta = phi = E = m = 0.0;
	p[0] = p[1] = p[2] = p[3] = 0.0;
}

Particle::Particle(Float_t px, Float_t py, Float_t pz, Float_t energy){ 

	p[0] = energy;
	p[1] = px;
	p[2] = py;
	p[3] = pz;
	
	// Computing rest of quantities

	Float_t m2 = energy*energy - (px*px + py*py + pz*pz);
	this->setMass(std::sqrt(m2));

	E = energy;
	pt = std::sqrt(px*px + py*pz);
	phi = std::atan2(py,px);
	eta = 0.5 * std::log((energy + pz)/(energy - pz));

}

Float_t Particle::sintheta() {
	Float_t p_magnitude = std::sqrt(p[1]*p[1] + p[2]*p[2] + p[3]*p[3]);
	Float_t sintheta = pt / p_magnitude;
	return sintheta;
}

void Particle::p4(Float_t pT, Float_t eta, Float_t phi, Float_t energy){

	Float_t m2 = energy*energy - pT*pT * std::pow(std::cosh(eta),2);
	this->setMass(std::sqrt(m2));
	this->E = energy;
	this->pt = pT;
	this->eta = eta;
	this->phi = std::fmod(phi, M_PI);
	
	Float_t px = pT * std::cos(phi);
	Float_t py = pT * std::sin(phi);
	Float_t pz = pT * std::sinh(eta);

	p[0] = energy;
	p[1] = px;
	p[2] = py;
	p[3] = pz;

}

void Particle::print(){
	std::cout << std::endl;
	std::cout << "p4 = (" << p[0] <<",\t" << p[1] <<",\t"<< p[2] <<",\t"<< p[3] << ")" << std::endl;
	std::cout << "sin(theta) = " << sintheta() << std::endl;
	// Printing out the other members
	std::cout << "pt: " << pt << ", eta: " << eta << ", phi: " << phi << ", E: " << E << ", m: " << m << std::endl;
}

//*****************************************************************************
//                                                                             *
//    MEMBERS functions of the Lepton Class                                    *
//                                                                             *
//*****************************************************************************


Lepton Lepton::FromPtEtaPhiE(Float_t pt, Float_t eta, Float_t phi, Float_t energy, Int_t charge) {
	Lepton obj;
	obj.p4(pt, eta, phi, energy);
	obj.setCharge(charge);
	return obj;
}

//*****************************************************************************
//                                                                             *
//    MEMBERS functions of the Jet Class                                       *
//                                                                             *
//*****************************************************************************


Jet Jet::FromPtEtaPhiE(Float_t pt, Float_t eta, Float_t phi, Float_t energy, Int_t hadronFlavor) {
	Jet obj; // Runs Particle constructor
	obj.p4(pt, eta, phi, energy);
	obj.setHadronFlavor(hadronFlavor);
	return obj;
}

//****************************************************************************

Int_t main() {
	
	/* ************* */
	/* Input Tree   */
	/* ************* */

	std::cout << "Program start!" << std::endl;

	TFile *f  = new TFile("input.root","READ");
	TTree *t1 = (TTree*)(f->Get("t1"));

	// Read the variables from the ROOT tree branches
	t1->SetBranchAddress("lepPt",&lepPt);
	t1->SetBranchAddress("lepEta",&lepEta);
	t1->SetBranchAddress("lepPhi",&lepPhi);
	t1->SetBranchAddress("lepE",&lepE);
	t1->SetBranchAddress("lepQ",&lepQ);
	
	t1->SetBranchAddress("njets",&njets);
	t1->SetBranchAddress("jetPt",&jetPt);
	t1->SetBranchAddress("jetEta",&jetEta);
	t1->SetBranchAddress("jetPhi",&jetPhi);
	t1->SetBranchAddress("jetE", &jetE);
	t1->SetBranchAddress("jetHadronFlavour",&jetHadronFlavour);

	// Total number of events in ROOT tree
	Long64_t nentries = t1->GetEntries();
	std::cout << "Number of entries: " << nentries << std::endl;

	for (Long64_t jentry=0; jentry<(Long64_t)nentries/1; jentry++)
 	{
		t1->GetEntry(jentry);
		std::cout<<"EVENT "<< jentry <<std::endl;	

		std::cout << "PRINTING LEPTON KINEMATICS: " << std::endl;
		for (Long64_t ilep=0; ilep<maxLepSize; ilep++) {
			Lepton lepton = Lepton::FromPtEtaPhiE(lepPt[ilep], lepEta[ilep], lepPhi[ilep], lepE[ilep], lepQ[ilep]);
			if (lepton.E < 1e-9) continue; // skip uninitialized leptons
			lepton.print();
			std::cout << "Charge: " << lepQ[ilep] << std::endl;
		}

		std::cout << std::endl;

		std::cout << "PRINTING JET KINEMATICS: " << std::endl;
		for (Long64_t ijet=0; ijet<maxJetSize; ijet++) {
			Particle jet = Jet::FromPtEtaPhiE(jetPt[ijet], jetEta[ijet], jetPhi[ijet], jetE[ijet], jetHadronFlavour[ijet]);
			if (jet.E < 1e-9) continue; // skip uninitialized jets
			jet.print();
			std::cout << "Hadron Flavor: " << jetHadronFlavour[ijet] << std::endl;
		}
		std::cout << "---------------------------------------------------------------" << std::endl;
	} // Loop over all events

  	return 0;
}
